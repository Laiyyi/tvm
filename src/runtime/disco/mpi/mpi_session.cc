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
#include <unistd.h>
#include <thread>


#include "../bcast_session.h"
#include "../message_queue.h"
#include "../disco_worker_thread.h"
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

class MPISessionObj final : public BcastSessionObj {
 public:
  explicit MPISessionObj(int num_workers)  : num_workers_(num_workers) {

  MPI_CALL(MPI_Init(nullptr, nullptr));
  const auto f_create_local_session =
        tvm::ffi::Function::GetGlobal("runtime.disco.create_mpi_session_local_workers");
  ICHECK(f_create_local_session.has_value())
        << "Cannot find function runtime.disco.create_mpi_session_local_workers";
  local_session_ = ((*f_create_local_session)(num_workers)).cast<BcastSession>();

  // MPI_Comm parent;
  // MPI_Comm_get_parent(&parent);

  // if( parent == MPI_COMM_NULL ) {

  
  //   const char* worker_command = "python3";
  //   char* worker_argv[] = {(char*)ex_name.c_str(), nullptr};

  //   MPI_Comm_spawn(
  //     worker_command, 
  //     worker_argv, 
  //     num_workers_,
  //     MPI_INFO_NULL, 0, 
  //     MPI_COMM_WORLD, 
  //     &intercomm,
  //     MPI_ERRCODES_IGNORE);
      
  //   MPI_Comm_rank(MPI_COMM_WORLD, &local_rank);
  //   MPI_Barrier(intercomm);
    
  //   std::cout << "[Master] Creating disco worker 0" << std::endl;
  //   worker_0_ = std::make_unique<DiscoWorkerThread>(0, num_workers, num_groups_, &worker_zero_data_);
  //   std::cout << "[Master] Disco worker 0 work !" << std::endl;

  //   // local_session_ = this.cast<BcastSession>();
  //   // DRef f_init_workers = local_session_->GetGlobalFunc("runtime.disco.mpi_session_init_workers");
  //   // local_session_->CallPacked(f_init_workers, num_workers, local_rank, 1 /*num_groups = 1*/, world_rank);

  //   //  int other_workers = num_workers -1;
  //   //  int rank = 1;
  //   //  for (int i = 0; i < other_workers; i++) {
  //   //       MPIRankStream stream(remote_rank);
  //   //       stream_.push_back(stream);
  //   //       channels.emplace_back(std::make_unique<DiscoMPIChannel>(stream_.back()));
  //   //       rank++;
  //   //   }

    
  // } else {

  //   MPI_Comm_rank(MPI_COMM_WORLD, &local_rank);

  //   int pipefd1[2]; // pipefd1[0] = main_read, pipefd1[1] = worker_write
  //   int pipefd2[2]; // pipefd2[0] = worker_read, pipefd2[1] = main_write
  //   pipe(pipefd1);
  //   pipe(pipefd2);
  //   std::string cmd = "python3 -m tvm.exec.disco_worker "
  //                     + std::to_string(local_rank + 1) + " "
  //                     + std::to_string(num_workers_-1) + " "
  //                     + std::to_string(num_groups_) + " "
  //                     + std::to_string(pipefd2[0]) + " "
  //                     + std::to_string(pipefd1[1]);


  //   worker_thread = std::thread([cmd](){
  //     system(cmd.c_str());
  //   });
  //   std::cout << "[Worker] MPI child process  " << local_rank << 
  //                " start disco worker, id = " << (local_rank+1) << " start!" << std::endl;
    
  //   MPI_Barrier(parent);
  //   to_master_channel = std::make_unique<DiscoMPIChannel>(to_master_stream);
    
    
  //   worker_thread.join();
  //   close(pipefd1[0]); close(pipefd1[1]);
  //   close(pipefd2[0]); close(pipefd2[1]);
  }



  //  if ( world_rank >= num_workers_per_node ) {
  //       local_channel = std::make_unique<DiscoMPIChannel>(localStream);
  //  }
    //  if ( world_rank != 0) { this->MainLoop(); }  
  }

  ~MPISessionObj() { Kill();    std::cout << "End" << std::endl;  }

  void Kill() {
   // 1. 取得 Parent 狀態
    MPI_Comm parent_comm;
    MPI_Comm_get_parent(&parent_comm);

    if (parent_comm == MPI_COMM_NULL) {
        // --- Master 邏輯 ---
        if (worker_0_ != nullptr) {
  
            this->Shutdown(); // 通知本地 worker 0 結束
            worker_0_.reset();
            printf("p ok!\n");
        }
        // 這裡應該送出一個信號給 intercomm，通知所有子進程結束
    } else {
        // --- Child 邏輯 ---
        // 確保 python worker 已經結束 (這裡要解決 system() 阻塞問題)
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
    }

    // 2. 移除 MPI_Barrier(MPI_COMM_WORLD)，直接 Finalize
    // 如果一定要同步，請確保是對著正確的 communicator
    MPI_Finalize();
  }

  int64_t GetNumWorkers() final { return num_workers_; }

  ffi::Any DebugGetFromRemote(int64_t reg_id, int worker_id) final {
    // int node_id = worker_id / num_workers_per_node;
    // if (node_id == 0) {
    //   return local_session_->DebugGetFromRemote(reg_id, worker_id);
    // } else {
      AnyView packed_args[5];
      ffi::PackedArgs::Fill(packed_args, static_cast<int>(DiscoMPIAction::kSend), worker_id,
                            static_cast<int>(DiscoAction::kDebugGetFromRemote), reg_id, worker_id);
      channels[worker_id + 1]->Send(ffi::PackedArgs(packed_args, 5));
      ffi::PackedArgs args = this->RecvReplyPacked(worker_id);
      ICHECK_EQ(args.size(), 2);
      ICHECK(static_cast<DiscoAction>(args[0].cast<int>()) == DiscoAction::kDebugGetFromRemote);
      ffi::Any result;
      result = args[1];
      return result;
    // }
  }
  

  void DebugSetRegister(int64_t reg_id, AnyView value, int worker_id) final { 
    // int node_id = worker_id / num_workers_per_node;
    // if (node_id == 0) {
    //   local_session_->DebugSetRegister(reg_id, value, worker_id);
    // } else {
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
        channels[worker_id + 1]->Send(ffi::PackedArgs(packed_args, 6));
      }
      ffi::Any result;
      ffi::PackedArgs args = this->RecvReplyPacked(worker_id);
      ICHECK_EQ(args.size(), 1);
      ICHECK(static_cast<DiscoAction>(args[0].cast<int>()) == DiscoAction::kDebugSetRegister);
    // }
  }

  void BroadcastPacked(const ffi::PackedArgs& args) override {
    // local_session_->BroadcastPacked(args);
    std::vector<AnyView> packed_args(args.size() + 2);
    ffi::PackedArgs::Fill(packed_args.data(), static_cast<int>(DiscoMPIAction::kSend), -1);
    std::copy(args.data(), args.data() + args.size(), packed_args.begin() + 2);
    for (auto& channel : channels) {

      channel->Send(ffi::PackedArgs(packed_args.data(), packed_args.size()));
    }
  }

  void SendPacked(int worker_id,
                  const ffi::PackedArgs& args) override {
    // int node_id = worker_id / num_workers_per_node;
    // if (node_id == 0) {
    //   local_session_->SendPacked(worker_id, args);
    //   return;
    // }
    std::vector<AnyView> packed_args(args.size() + 2);
    ffi::PackedArgs::Fill(packed_args.data(), static_cast<int>(DiscoMPIAction::kSend),
                          worker_id);
    std::copy(args.data(), args.data() + args.size(), packed_args.begin() + 2);
    channels[worker_id + 1]->Send(ffi::PackedArgs(packed_args.data(), packed_args.size()));
  }



  ffi::PackedArgs RecvReplyPacked(int worker_id) override {
    // int node_id = worker_id / num_workers_per_node;
    // if (node_id == 0) {
    //   return local_session_->RecvReplyPacked(worker_id);
    // }
    AnyView packed_args[2];
    ffi::PackedArgs::Fill(packed_args, static_cast<int>(DiscoMPIAction::kReceive), worker_id);
    channels[worker_id + 1]->Send(ffi::PackedArgs(packed_args, 2));
    return channels[worker_id + 1]->Recv();

  }

  //  void MainLoop() {
  //   while (true) {
  //     ffi::PackedArgs args = to_master_channel->Recv();
  //     DiscoMPIAction action = static_cast<DiscoMPIAction>(args[0].cast<int>());
  //     int worker_id = args[1].cast<int>();
  //     int local_worker_id = this->local_rank;
  //     switch (action) {
  //       case DiscoMPIAction::kSend: {
  //         args = args.Slice(2);
  //         if (worker_id == -1) {
  //           local_session_->BroadcastPacked(args);
  //         } else {
  //           local_session_->SendPacked(local_worker_id, args);
  //         }
  //         break;
  //       }
  //       case DiscoMPIAction::kReceive: {
  //         args = local_session_->RecvReplyPacked(local_worker_id);
  //         to_master_channel->Reply(args);
  //         break;
  //       }
  //       case DiscoMPIAction::kShutdown: {
  //         local_session_->Shutdown();
  //         return;
  //       }
  //       default:
  //         LOG(FATAL) << "Invalid action " << static_cast<int>(action);
  //     }
  //   }
  // }

  // void Shutdown() final {
  //   // local session will be implicitly shutdown by its destructor
  //   std::vector<AnyView> packed_args(2);
  //   ffi::PackedArgs::Fill(packed_args.data(), static_cast<int>(DiscoMPIAction::kShutdown), -1);
  //   for (auto& channel : remote_channels) {
  //     channel->Send(ffi::PackedArgs(packed_args.data(), packed_args.size()));
  //   }
  //   remote_channels.clear();
  //   MPI_CALL(MPI_Finalize());
  // }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("runtime.disco.MPISession", MPISessionObj,BcastSessionObj);
  
  int num_workers_;
  int num_groups_ = 1;
  const char* ex_name_;
  MPI_Comm intercomm;
  std::thread worker_thread;

  BcastSession local_session_{nullptr};

std::unique_ptr<DiscoWorkerThread> worker_0_  = nullptr;
  std::vector<MPIRankStream> stream_;
  std::vector<std::unique_ptr<DiscoMPIChannel>> channels;
  //other ranks use.

  MPIRankStream to_master_stream{0};
  std::unique_ptr<DiscoMPIChannel> to_master_channel;

  int num_workers_per_node;

  int local_rank;
  int world_rank;

};


Session MPISession( int num_workers ) {
  auto n = ffi::make_object<MPISessionObj>( num_workers );
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
