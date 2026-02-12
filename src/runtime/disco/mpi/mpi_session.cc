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
  LOG(INFO) << "Number of processes on this node: " << num_workers_per_node 
            << ". Initializing worker group with " << num_workers << " mpi processes.";
  
  DRef f_init_workers =
        local_session_->GetGlobalFunc("runtime.disco.mpi_session_init_workers");
    local_session_->CallPacked(f_init_workers, num_workers, local_rank, 1 /*num_groups = 1*/, num_workers_per_node, world_rank);
  
}

~MPISessionObj() {
  MPI_CALL(MPI_Finalize());
}

 ffi::Any DebugGetFromRemote(int64_t reg_id, int worker_id) final {
      ffi::Any result;

      return result;
    }
  

  void DebugSetRegister(int64_t reg_id, AnyView value, int worker_id) final {

  }
  int64_t GetNumWorkers() final { return num_workers; }

  void BroadcastPacked(const ffi::PackedArgs& args) override {
      // 用 MPI_Bcast 實作
  }

  void SendPacked(int worker_id,
                  const ffi::PackedArgs& args) override {
      // 用 MPI_Send
  }

  ffi::PackedArgs RecvReplyPacked(int worker_id) override {
      // 用 MPI_Recv
      ffi::PackedArgs args = this->RecvReplyPacked(worker_id);
      return args;
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("runtime.disco.MPISession", MPISessionObj,BcastSessionObj);
  int num_workers_per_node;
  int num_workers;
  int local_rank;
  int world_rank;

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
           [](int num_workers, int local_rank, int num_groups, int num_workers_per_node, int world_rank) {
             LOG(INFO) << "Initializing local worker  : " << local_rank << " & it's world worker id : " << world_rank
                       << " with " << num_groups << " groups.";
  
             DiscoWorker* worker = DiscoWorker::ThreadLocal();
             worker->local_worker_id = local_rank;
             worker->num_groups = num_groups;
             worker->worker_id = world_rank;
             worker->num_workers = num_workers;
           });
}

}  // namespace runtime
}  // namespace tvm
