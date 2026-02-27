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

class DiscoMPI : public dmlc::Stream {
  
  size_t RecvAll(void* buf_, size_t len, int src_rank, int tag, MPI_Comm comm) {
    
    MPI_Status status;
    MPI_CALL(MPI_Recv(buf_, static_cast<int>(len), MPI_BYTE, src_rank, tag, comm, &status ));

    int actual = 0;
    MPI_CALL(MPI_Get_count(&status, MPI_BYTE, &actual));

    return static_cast<size_t>(actual);
  }

  size_t SendAll(const void* buf_, size_t len, int dst_rank, int tag, MPI_Comm comm) {

     MPI_CALL(MPI_Send(buf_, static_cast<int>(len), MPI_BYTE, dst_rank, tag, comm));
     return len;  
  }
  size_t Read(void* data, size_t size) final { return RecvAll(data, size); }

  size_t Write(const void* data, size_t size) final { return SendAll(data, size); }
}



class DiscoMPIChannel : public DiscoChannel {
 public:
  explicit DiscoMPIChannel(const DiscoMPI& mesg_)
      : mesg(mesg_), message_queue_(&mesg_) {}

  DiscoMPIChannel(DiscoMPIChannel&& other) = delete;
  DiscoMPIChannel(const DiscoMPIChannel& other) = delete;
  void Send(const ffi::PackedArgs& args) { message_queue_.Send(args); }
  ffi::PackedArgs Recv() { return message_queue_.Recv(); }
  void Reply(const ffi::PackedArgs& args) { message_queue_.Send(args); }
  ffi::PackedArgs RecvReply() { return message_queue_.Recv(); }

 private:
  DiscoMPI mesg;
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
    
   int num_nodes = num_workers / num_workers_per_node;

   for (int i = 0; i + 1 < num_nodes; ++i) {
     remote_channels.emplace_back(std::make_unique<DiscoMPIChannel>); 
    }
  }

  ~MPISessionObj() { MPI_CALL(MPI_Finalize());}

  int64_t GetNumWorkers() final { return num_workers; }

  ffi::Any DebugGetFromRemote(int64_t reg_id, int worker_id) final {
    int node_id = worker_id / num_workers_per_node;
    if (node_id == 0) {
      return local_session_->DebugGetFromRemote(reg_id, worker_id);
    } else {
      AnyView packed_args[5];
      ffi::PackedArgs::Fill(packed_args, static_cast<int>(DiscoSocketAction::kSend), worker_id,
                            static_cast<int>(DiscoAction::kDebugGetFromRemote), reg_id, worker_id);
      remote_channels[node_id - 1]->Send(ffi::PackedArgs(packed_args, 5));
      ffi::PackedArgs args = this->RecvReplyPacked(worker_id);
      ICHECK_EQ(args.size(), 2);
      ICHECK(static_cast<DiscoAction>(args[0].cast<int>()) == DiscoAction::kDebugGetFromRemote);
      ffi::Any result;
      result = args[1];
      return result;
    }
  }
  

  void DebugSetRegister(int64_t reg_id, AnyView value, int worker_id) final { 
       int node_id = worker_id / num_workers_per_node_;
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
        ffi::PackedArgs::Fill(packed_args, static_cast<int>(DiscoSocketAction::kSend), worker_id,
                              static_cast<int>(DiscoAction::kDebugSetRegister), reg_id, worker_id,
                              value);
        remote_channels_[node_id - 1]->Send(ffi::PackedArgs(packed_args, 6));
      }
      ffi::Any result;
      ffi::PackedArgs args = this->RecvReplyPacked(worker_id);
      ICHECK_EQ(args.size(), 1);
      ICHECK(static_cast<DiscoAction>(args[0].cast<int>()) == DiscoAction::kDebugSetRegister);
    }
  }

  void BroadcastPacked(const ffi::PackedArgs& args) override {

    // MPI_CALL(MPI_Bcast(args.data(), args.size(), MPI_INT, 0, MPI_COMM_WORLD));
       local_session_->BroadcastPacked(args);
    std::vector<AnyView> packed_args(args.size() + 2);
    ffi::PackedArgs::Fill(packed_args.data(), static_cast<int>(DiscoSocketAction::kSend), -1);
    std::copy(args.data(), args.data() + args.size(), packed_args.begin() + 2);
    for (auto& channel : remote_channels_) {
      channel->Send(ffi::PackedArgs(packed_args.data(), packed_args.size()));
    }
  }

  void SendPacked(int worker_id,
                  const ffi::PackedArgs& args) override {
      // 用 MPI_Send

          int node_id = worker_id / num_workers_per_node_;
    if (node_id == 0) {
      local_session_->SendPacked(worker_id, args);
      return;
    }
    std::vector<AnyView> packed_args(args.size() + 2);
    ffi::PackedArgs::Fill(packed_args.data(), static_cast<int>(DiscoSocketAction::kSend),
                          worker_id);
    std::copy(args.data(), args.data() + args.size(), packed_args.begin() + 2);
    remote_channels_[node_id - 1]->Send(ffi::PackedArgs(packed_args.data(), packed_args.size()));
  }


  /*!
   * \brief Receive a packed sequence from a worker. This function is usually called by the
   * controler to communicate with worker-0, because the worker-0 is assumed to be always
   collocated
   * with the controler. Receiving from other workers may not be supported.
   * \return The packed sequence received.
   */

  ffi::PackedArgs RecvReplyPacked(int worker_id) override {
    // int node_id = worker_id / num_workers_per_node;
    // if (node_id == 0) {
    //   return local_session_->RecvReplyPacked(worker_id);
    // }
    // int size;
    // OMPI_DECLSPEC  int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source,
    //                         int tag, MPI_Comm comm, MPI_Status *status);
    // MPI_Recv(&size, 1, MPI_INT, worker_id, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // std::vector<char> buffer(size);
    // MPI_Recv(buffer.data(), size, MPI_BYTE, worker_id, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // ffi::PackedArgs args = DeserializePacked(buffer);
  
    AnyView* data;
    int size = 0;
    ffi::PackedArgs args(data,size);


      int node_id = worker_id / num_workers_per_node_;
    if (node_id == 0) {
      return local_session_->RecvReplyPacked(worker_id);
    }
    AnyView packed_args[2];
    ffi::PackedArgs::Fill(packed_args, static_cast<int>(DiscoSocketAction::kReceive), worker_id);
    remote_channels_[node_id - 1]->Send(ffi::PackedArgs(packed_args, 2));
    return remote_channels_[node_id - 1]->Recv();
    return args;
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("runtime.disco.MPISession", MPISessionObj,BcastSessionObj);
  int num_workers_per_node;
  int num_workers;
  int local_rank;
  int world_rank;
  std::vector<std::unique_ptr<DiscoMPIChannel>> remote_channels;
  BcastSession local_session_{nullptr};
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
