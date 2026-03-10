/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include "../bcast_session.h"
#include "../message_queue.h"
#include "mpi_context.h"

namespace tvm {
namespace runtime {

enum class DiscoMPIAction {
  kShutdown = static_cast<int>(DiscoAction::kShutDown),
  kSend,
  kReceive,
};

class MPIRankStream : public dmlc::Stream {
  public:
   explicit MPIRankStream(int world_rank)
       : peer_rank_(world_rank) {}
  
  size_t RecvAll(void* buf_, size_t len) { 
    MPI_Status status;
    // receive data from peer_rank
    MPI_CALL(MPI_Recv(buf_, static_cast<int>(len), MPI_BYTE, this->peer_rank_, static_cast<int>(DiscoMPIAction::kReceive), MPI_COMM_WORLD, &status));

    int actual = 0;
    MPI_CALL(MPI_Get_count(&status, MPI_BYTE, &actual));

    return static_cast<size_t>(actual);
  }

  size_t SendAll(const void* buf_, size_t len) {
     //send data to peer_rank
     MPI_CALL(MPI_Send(buf_, static_cast<int>(len), MPI_BYTE, this->peer_rank_, static_cast<int>(DiscoMPIAction::kSend), MPI_COMM_WORLD));
     return len;  
  }
  size_t Read(void* data, size_t size) final { return RecvAll(data, size); }

  size_t Write(const void* data, size_t size) final { return SendAll(data, size); }

  private:
    int peer_rank_{-1};  
};

class DiscoMPIChannel : public DiscoChannel {
 public:
  explicit DiscoMPIChannel(const MPIRankStream& transport)
      : transport_(transport), message_queue_(&transport_) {}

  DiscoMPIChannel(DiscoMPIChannel&& other) = delete;
  DiscoMPIChannel(const DiscoMPIChannel& other) = delete;
  void Send(const ffi::PackedArgs& args) { message_queue_.Send(args); }
  ffi::PackedArgs Recv() { return message_queue_.Recv(); }
  void Reply(const ffi::PackedArgs& args) { message_queue_.Send(args); }
  ffi::PackedArgs RecvReply() { return message_queue_.Recv(); }

 private:
  MPIRankStream transport_;
  DiscoStreamMessageQueue message_queue_;
};

class MPISessionObj : public BcastSessionObj {
 public:
  explicit MPISessionObj() {

   // Reuse create_socket_session_local_workers to create local session.
   const auto f_create_local_session =
        tvm::ffi::Function::GetGlobal("runtime.disco.create_socket_session_local_workers");
   ICHECK(f_create_local_session.has_value())
        << "Cannot find function runtime.disco.create_socket_session_local_workers";
   local_session_ = ((*f_create_local_session)(1)).cast<BcastSession>();

   MPI_CALL(MPI_Init(nullptr, nullptr));
   MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_workers));
   MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));

   MPI_Comm local_comm;
   MPI_CALL(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm));
   MPI_CALL(MPI_Comm_rank(local_comm, &local_rank));
   MPI_CALL(MPI_Comm_size(local_comm, &num_workers_per_node));
   MPI_CALL(MPI_Comm_free(&local_comm));
   DRef f_init_workers =
        local_session_->GetGlobalFunc("runtime.disco.mpi_session_init_workers");
   local_session_->CallPacked(f_init_workers, num_workers, local_rank, 1 /*num_groups = 1*/, world_rank);
    

   
   
   // only root need to create channel
   if ( world_rank == 0 ) {
      int remotes_workers = num_workers - num_workers_per_node;
      int remote_rank = num_workers_per_node;
     
      for (int i = 0; i < remotes_workers; i++) {
         MPIRankStream stream(remote_rank);
         remote_stream.push_back(stream);
         remote_channels.emplace_back(std::make_unique<DiscoMPIChannel>(remote_stream.back()));
         remote_rank++;
      }
   } else if ( world_rank >= num_workers_per_node ) {
        local_channel = std::make_unique<DiscoMPIChannel>(localStream);
   }
   
   if ( world_rank != 0) { this->MainLoop(); }  
   

  }

  ~MPISessionObj() { MPI_CALL(MPI_Finalize());}

  int64_t GetNumWorkers() final { return num_workers; }

  ffi::Any DebugGetFromRemote(int64_t reg_id, int worker_id) final {
    int node_id = worker_id / num_workers_per_node;
    if (node_id == 0) {
      return local_session_->DebugGetFromRemote(reg_id, worker_id);
    } else {
      AnyView packed_args[5];
      ffi::PackedArgs::Fill(packed_args, static_cast<int>(DiscoMPIAction::kSend), worker_id,
                            static_cast<int>(DiscoAction::kDebugGetFromRemote), reg_id, worker_id);
      remote_channels[worker_id - num_workers_per_node]->Send(ffi::PackedArgs(packed_args, 5));
      ffi::PackedArgs args = this->RecvReplyPacked(worker_id);
      ICHECK_EQ(args.size(), 2);
      ICHECK(static_cast<DiscoAction>(args[0].cast<int>()) == DiscoAction::kDebugGetFromRemote);
      ffi::Any result;
      result = args[1];
      return result;
    }
  }
  

  void DebugSetRegister(int64_t reg_id, AnyView value, int worker_id) final { 
       int node_id = worker_id / num_workers_per_node;
    if (node_id == 0) {
      local_session_->DebugSetRegister(reg_id, value, worker_id);
    } else {
      ObjectRef wrapped{nullptr};
      if (auto opt_obj = value.as<ObjectRef>()) {
        wrapped = DiscoDebugObject::Wrap(value);
        value = wrapped;
      }
      {
        AnyView packed_args[6];
        ffi::PackedArgs::Fill(packed_args, static_cast<int>(DiscoMPIAction::kSend), worker_id,
                              static_cast<int>(DiscoAction::kDebugSetRegister), reg_id, worker_id,
                              value);
        remote_channels[worker_id - num_workers_per_node]->Send(ffi::PackedArgs(packed_args, 6));
      }
      ffi::Any result;
      ffi::PackedArgs args = this->RecvReplyPacked(worker_id);
      ICHECK_EQ(args.size(), 1);
      ICHECK(static_cast<DiscoAction>(args[0].cast<int>()) == DiscoAction::kDebugSetRegister);
    }
  }

  void BroadcastPacked(const ffi::PackedArgs& args) override {
    local_session_->BroadcastPacked(args);
    std::vector<AnyView> packed_args(args.size() + 2);
    ffi::PackedArgs::Fill(packed_args.data(), static_cast<int>(DiscoMPIAction::kSend), -1);
    std::copy(args.data(), args.data() + args.size(), packed_args.begin() + 2);
    for (auto& channel : remote_channels) {
      channel->Send(ffi::PackedArgs(packed_args.data(), packed_args.size()));
    }
  }

  void SendPacked(int worker_id,
                  const ffi::PackedArgs& args) override {
    int node_id = worker_id / num_workers_per_node;
    if (node_id == 0) {
      local_session_->SendPacked(worker_id, args);
      return;
    }
    std::vector<AnyView> packed_args(args.size() + 2);
    ffi::PackedArgs::Fill(packed_args.data(), static_cast<int>(DiscoMPIAction::kSend),
                          worker_id);
    std::copy(args.data(), args.data() + args.size(), packed_args.begin() + 2);
    remote_channels[worker_id - num_workers_per_node]->Send(ffi::PackedArgs(packed_args.data(), packed_args.size()));
  }



  ffi::PackedArgs RecvReplyPacked(int worker_id) override {
    int node_id = worker_id / num_workers_per_node;
    if (node_id == 0) {
      return local_session_->RecvReplyPacked(worker_id);
    }
    AnyView packed_args[2];
    ffi::PackedArgs::Fill(packed_args, static_cast<int>(DiscoMPIAction::kReceive), worker_id);
    remote_channels[worker_id - num_workers_per_node]->Send(ffi::PackedArgs(packed_args, 2));
    return remote_channels[worker_id - num_workers_per_node]->Recv();

  }

   void MainLoop() {
    while (true) {
      ffi::PackedArgs args = local_channel->Recv();
      DiscoMPIAction action = static_cast<DiscoMPIAction>(args[0].cast<int>());
      int worker_id = args[1].cast<int>();
      int local_worker_id = this->local_rank;
      switch (action) {
        case DiscoMPIAction::kSend: {
          args = args.Slice(2);
          if (worker_id == -1) {
            local_session_->BroadcastPacked(args);
          } else {
            local_session_->SendPacked(local_worker_id, args);
          }
          break;
        }
        case DiscoMPIAction::kReceive: {
          args = local_session_->RecvReplyPacked(local_worker_id);
          local_channel->Reply(args);
          break;
        }
        case DiscoMPIAction::kShutdown: {
          local_session_->Shutdown();
          return;
        }
        default:
          LOG(FATAL) << "Invalid action " << static_cast<int>(action);
      }
    }
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("runtime.disco.MPISession", MPISessionObj,BcastSessionObj);
  
  int num_workers_per_node;
  int num_workers;
  int local_rank;
  int world_rank;
  BcastSession local_session_{nullptr};

  //only rank = 0 use.
  std::vector<MPIRankStream> remote_stream;
  std::vector<std::unique_ptr<DiscoMPIChannel>> remote_channels;
  //other ranks use.
  MPIRankStream localStream{0};
  std::unique_ptr<DiscoMPIChannel> local_channel;
};


Session MPISession() {
  auto n = ffi::make_object<MPISessionObj>();
  return Session(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<MPISessionObj>();
  refl::GlobalDef()
      .def("runtime.disco.MPISession", MPISession)
      .def("runtime.disco.mpi_session_init_workers",
           [](int num_workers, int local_rank, int num_groups, int world_rank) {
             DiscoWorker* worker = DiscoWorker::ThreadLocal();
             LOG(INFO) << "Initializing local worker  : " << local_rank << " & it's world worker id : " << world_rank
                       << " with " << num_groups << " groups.";  
             worker->local_worker_id = local_rank;  
             worker->num_groups = num_groups;
             worker->worker_id = world_rank;
             worker->num_workers = num_workers;

           });
}

}  // namespace runtime
}  // namespace tvm
