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

#include <tvm/ffi/reflection/registry.h>

#include <cstring>
#include <mutex>
#include <sstream>
#include <vector>

#include "../../../support/process_id.h"
#include "../utils.h"
#include "mpi_context.h"
#include <climits>
#include <fstream>

namespace tvm {
namespace runtime {
namespace mpi {

CCLThreadLocalContext* CCLThreadLocalContext::Get() {
  thread_local static CCLThreadLocalContext ctx;
  return &ctx;
}

inline MPI_Op AsMPIRedOp(ReduceKind kind) {
    switch (kind) {
        case ReduceKind::kSum:
            return MPI_SUM;
        case ReduceKind::kProd:
            return MPI_PROD;
        case ReduceKind::kMin:
            return MPI_MIN;
        case ReduceKind::kMax:
            return MPI_MAX;
        case ReduceKind::kAvg:
            // MPI 沒有內建 AVG，需要自己計算 sum/count
            LOG(FATAL) << "ReduceKind::kAvg not directly supported in MPI";
    }
    LOG(FATAL) << "ValueError: Unknown ReduceKind: " << static_cast<int>(kind);
    throw ;
    
}



void InitCCL(Session sess, ffi::Shape device_ids) {
  DRef func = sess->GetGlobalFunc("runtime.disco." TVM_DISCO_CCL_NAME ".init_ccl_per_worker");
  DLOG(INFO) << "Initializing " TVM_DISCO_CCL_NAME " with devices: " << device_ids;
  sess->CallPacked(func, device_ids);
}

void InitCCLPerWorker(ffi::Shape device_ids) {
   
  CCLThreadLocalContext* ctx = CCLThreadLocalContext::Get();
  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  ICHECK(worker != nullptr);

  CHECK(!ctx->worker) << "Cannot initialize OpenMPI, "
                      << "the previous thread-global worker still exists, "
                      << "and has not been destructed";

  // Step up local context of OpenMPI
  // Assuming that the number of workers per node is uniform
  int group_size = worker->num_workers;
  int num_workers_per_node = group_size / device_ids.size();
  int index = worker->worker_id / num_workers_per_node;
  int device_id = device_ids[index];

  Device device{TVM_DISCO_DEVICE_TYPE, device_id};
    if (worker->default_device.device_type == DLDeviceType::kDLCPU) {
    worker->default_device = device;
  } else {
    ICHECK(worker->default_device.device_type == device.device_type &&
           worker->default_device.device_id == device.device_id)
        << "The default device of the worker is inconsistent with the device used for CCL. "
        << "The default device is " << worker->default_device << ", but the device used for CCL is "
        << device << ".";
  }
  worker->ccl = TVM_DISCO_CCL_NAME;
  ctx->worker = worker;
  ctx->device_id = device_id;
  
  // std::ofstream fout("debug_log.txt", std::ios::out);
  //   if (!fout.is_open()) {
  //       std::cerr << "Failed to open debug_log.txt" << std::endl;
   
  //   }

  //   // 寫入檔案
  //   fout << "group size = " << group_size << std::endl;
  //   fout << "worker->num_workers = " << worker->num_workers << std::endl;
  //   fout << "worker->num_groups = " << worker->num_groups << std::endl;
  //   fout << "device id = " << device_id << std::endl;
  //   fout << "worker->local_worker_id = " << worker->local_worker_id << std::endl;
  //   fout << "worker->ccl = " << worker->ccl << std::endl;
  //   fout << "ctx->worker = " << ctx->worker << std::endl;
  //   fout << "ctx->device_id = " << ctx->device_id << std::endl;
  //   fout << "ctx->worker->worker_id = " << ctx->worker->worker_id << std::endl;

  //   fout.close(); // 關檔
   

}

void AllReduce(Tensor send, ReduceKind reduce_kind, bool in_group, Tensor recv) {
  CCLThreadLocalContext* ctx = CCLThreadLocalContext::Get();
  ffi::Shape shape = send.Shape();
  int32_t numel = static_cast<int32_t>(shape->Product());

  DataType dtype = DataType(send->dtype);
  if (dtype == DataType::Float8E4M3FN() || dtype == DataType::Float8E5M2() || dtype == DataType::BFloat(16) ||
      dtype == DataType::UInt(8) || dtype == DataType::Float(16)) {
    LOG(FATAL) << "Unsupported data type for allreduce: MPI does not support this dtype.";
  }

  MPI_CALL(MPI_Allreduce(send->data, recv->data, numel,
                  /*datatype=*/AsMPIDataType(dtype),
                  /*op=*/AsMPIRedOp(reduce_kind),
                  MPI_COMM_WORLD));
 

}

void AllGather(Tensor send, bool in_group, Tensor recv) {
  CCLThreadLocalContext* ctx = CCLThreadLocalContext::Get();
  ffi::Shape shape = send.Shape();
  int32_t numel = static_cast<int32_t>(shape->Product());

  DataType dtype = DataType(send->dtype);
  if (dtype == DataType::Float8E4M3FN() || dtype == DataType::Float8E5M2() || dtype == DataType::BFloat(16) ||
      dtype == DataType::UInt(8) || dtype == DataType::Float(16)) {
    LOG(FATAL) << "Unsupported data type for allgather: MPI does not support this dtype.";
  }

    MPI_CALL(MPI_Allgather(
        send->data, numel,
        /*datatype=*/AsMPIDataType(dtype),
        recv->data, numel,
        /*datatype=*/AsMPIDataType(dtype),
        MPI_COMM_WORLD
     ));


 
}

void BroadcastFromWorker0(ffi::Optional<Tensor> send, bool in_group, Tensor recv) {
  CCLThreadLocalContext* ctx = CCLThreadLocalContext::Get();

  int worker_id = ctx->worker->worker_id;
  int group_size = ctx->worker->num_workers;
  bool is_sender = (worker_id == 0 && !in_group) || (in_group && worker_id % group_size == 0);
  
  int32_t numel = static_cast<int32_t>(recv.Shape().Product());

  DataType dtype = DataType(recv->dtype);
  int size;
  MPI_Type_size(AsMPIDataType(dtype), &size);

  if (is_sender) {
    CHECK(send.defined());
    CHECK(send.value().Shape().Product() == numel);
    std::memcpy(recv->data,
                send.value()->data,
                numel * size);
  } 

  void* chunk_ptr = static_cast<char*>(recv->data);

  MPI_CALL(MPI_Bcast(chunk_ptr, numel, AsMPIDataType(dtype),/*root=*/0, MPI_COMM_WORLD));


  
}

void ScatterFromWorker0(ffi::Optional<Tensor> send, bool in_group, Tensor recv) {
  CHECK(recv.defined()) << "ValueError: buffer `recv` must not be None";
  CCLThreadLocalContext* ctx = CCLThreadLocalContext::Get();
  int worker_id = ctx->worker->worker_id;
  int num_workers = ctx->worker->num_workers;
  int group_size = ctx->worker->num_workers;
  bool is_sender = (worker_id == 0 && !in_group) || (in_group && worker_id % group_size == 0);
  int num_receiver = in_group ? group_size : num_workers;
 

  int type_size;
  MPI_Type_size(AsMPIDataType(DataType(recv->dtype)), &type_size);

  int32_t numel_per_shard = static_cast<int32_t>(recv.Shape().Product());

  // ---------- Root 檢查 ----------
  if (is_sender) {
    CHECK(send.defined())
        << "ValueError: send must be provided when root.";

    Tensor buffer = send.value();
    int32_t total_numel = static_cast<int32_t>(buffer.Shape().Product());

    CHECK_EQ(total_numel % num_receiver, 0)
        << "Scatter requires total elements divisible by number of receivers.";

    CHECK_EQ(total_numel / num_receiver,
             numel_per_shard)
        << "recv size mismatch.";
  }

  void* recv_chunk = static_cast<char*>(recv->data);
  const void* send_chunk = nullptr;

    if (is_sender) {
      send_chunk =
          static_cast<char*>(send.value()->data) +
          (ctx->worker->worker_id * numel_per_shard) * type_size;
    }

    MPI_CALL(MPI_Scatter(
        send_chunk,        // root only meaningful
        numel_per_shard,             // sendcount per rank
        AsMPIDataType(DataType(recv->dtype)),
        recv_chunk,
        numel_per_shard,
        AsMPIDataType(DataType(recv->dtype)),
        /*root=*/0,
        MPI_COMM_WORLD));

}

// void GatherToWorker0(Tensor send, bool in_group, ffi::Optional<Tensor> recv) {
//   CHECK(send.defined()) << "ValueError: buffer `send` must not be None";
//   CCLThreadLocalContext* ctx = CCLThreadLocalContext::Get();
//   int worker_id = ctx->worker->worker_id;
//   int num_workers = ctx->worker->num_workers;
//   int group_size = num_workers;
//   bool is_sender = (worker_id == 0 && !in_group) || (in_group && worker_id % group_size == 0);
//   int num_receiver = in_group ? group_size : num_workers;

//   int64_t numel_send = send.Shape().Product();

//   MPI_Datatype dtype = AsMPIDataType(DataType(send->dtype));
//   int type_size;
//   MPI_Type_size(dtype, &type_size);

//   // ---------- root 檢查 ----------
//   if (is_sender) {
//     CHECK(recv.defined())
//         << "ValueError: recv buffer must be provided on root";

//     Tensor recv_buf = recv.value();
//     int64_t numel_recv = recv_buf.Shape().Product();

//     CHECK_EQ(numel_recv % num_receiver, 0)
//         << "Gather requires recv numel divisible by num_receiver";

//     int64_t numel_per_shard = numel_recv / num_receiver;
//     CHECK_EQ(numel_per_shard, numel_send)
//         << "send.size must match each shard in recv";
//   }

//   // ---------- int64 安全版本 ----------
//   int64_t offset = 0;
//   int64_t numel_per_shard = numel_send;

//   while (offset < numel_per_shard) {
//     int chunk = static_cast<int>(
//         std::min<int64_t>(INT_MAX, numel_per_shard - offset));

//     void* send_ptr =
//         static_cast<char*>(send->data) + offset * type_size;

//     void* recv_ptr = nullptr;
//     if (is_sender) {
//       recv_ptr =
//           static_cast<char*>(recv.value()->data) +
//           offset * type_size;
//     }

//     MPI_CALL(MPI_Gather(
//         send_ptr,
//         chunk,
//         dtype,
//         recv_ptr,   // only root meaningful
//         chunk,
//         dtype,
//         /*root=*/0,
//         MPI_COMM_WORLD));

//     offset += chunk;
//   }
// }

// void RecvFromWorker0(Tensor buffer) {
//   CCLThreadLocalContext* ctx = CCLThreadLocalContext::Get();
  
//   CHECK_NE(ctx->worker->worker_id, 0)
//       << "ValueError: Worker 0 is not allowed to call RecvFromWorker0.";
//   DataType dtype(buffer->dtype);
//   MPI_Datatype mpi_dtype = AsMPIDataType(dtype);

//   int type_size;
//   MPI_Type_size(mpi_dtype, &type_size);

//   int64_t numel = buffer.Shape().Product();

//   // ---------- int64 safe recv ----------
//   int64_t offset = 0;
//   while (offset < numel) {
//       int chunk = static_cast<int>(std::min<int64_t>(INT_MAX, numel - offset));
//       void* chunk_ptr = static_cast<char*>(buffer->data) + offset * type_size;

//         MPI_CALL(MPI_Recv(
//             chunk_ptr,
//             chunk,
//             mpi_dtype,
//             /*source=*/0,
//             /*tag=*/0,
//             MPI_COMM_WORLD,
//             MPI_STATUS_IGNORE));

//         offset += chunk;
//     }
// }

// void SendToNextGroup(Tensor buffer) {
//   CCLThreadLocalContext* ctx = CCLThreadLocalContext::Get();
  
//   int worker_id = ctx->worker->worker_id;
//   int group_size = ctx->worker->num_workers;
//   int receiver_id = worker_id + group_size;
//   int num_groups = ctx->worker->num_groups;
//   CHECK_LT(receiver_id, ctx->worker->num_workers)
//       << "The current group is already the last group and there is no such a next group.";

//   DataType dtype(buffer->dtype);
//   MPI_Datatype mpi_dtype = AsMPIDataType(dtype);

//     int type_size;
//     MPI_Type_size(mpi_dtype, &type_size);

//     int64_t numel = buffer.Shape().Product();

//     // ---------- int64-safe send ----------
//     int64_t offset = 0;
//     while (offset < numel) {
//         int chunk = static_cast<int>(std::min<int64_t>(INT_MAX, numel - offset));

//         void* chunk_ptr = static_cast<char*>(buffer->data) + offset * type_size;

//         MPI_CALL(MPI_Send(
//             chunk_ptr,
//             chunk,
//             mpi_dtype,
//             receiver_id,
//             /*tag=*/0,
//             MPI_COMM_WORLD));

//         offset += chunk;
//     }
  
// }

// void RecvFromPrevGroup(Tensor buffer) {
//     CCLThreadLocalContext* ctx = CCLThreadLocalContext::Get();
//     int worker_id = ctx->worker->worker_id;
//     int world_size = ctx->worker->num_workers;
//     int num_groups = ctx->worker->num_groups;

//     int group_size = world_size / num_groups;
//     int sender_id = worker_id - group_size;

//     CHECK_GE(sender_id, 0)
//         << "The current group is already the first group and there is no previous group.";

    
//     DataType dtype(buffer->dtype);
//     MPI_Datatype mpi_dtype = AsMPIDataType(dtype);

//     int type_size;
//     MPI_Type_size(mpi_dtype, &type_size);

//     int64_t numel = buffer.Shape().Product();

//     // ---------- int64-safe recv ----------
//     int64_t offset = 0;
//     while (offset < numel) {
//         int chunk = static_cast<int>(std::min<int64_t>(INT_MAX, numel - offset));

//         void* chunk_ptr = static_cast<char*>(buffer->data) + offset * type_size;

//         MPI_CALL(MPI_Recv(
//             chunk_ptr,
//             chunk,
//             mpi_dtype,
//             sender_id,
//             /*tag=*/0,
//             MPI_COMM_WORLD,
//             MPI_STATUS_IGNORE));

//         offset += chunk;
//     }
// }

// void SendToWorker(Tensor buffer, int receiver_id) {
//   CCLThreadLocalContext* ctx = CCLThreadLocalContext::Get();
//   int worker_id = ctx->worker->worker_id;
//   CHECK(receiver_id >= 0 && receiver_id < ctx->worker->num_workers)
//       << "Invalid receiver id " << receiver_id << ". The world size is "
//       << ctx->worker->num_workers;
//   CHECK_NE(worker_id, receiver_id) << "Cannot send to worker itself.";
//   DataType dtype(buffer->dtype);
//   MPI_Datatype mpi_dtype = AsMPIDataType(dtype);

//   int type_size;
//   MPI_Type_size(mpi_dtype, &type_size);

//     int64_t numel = buffer.Shape().Product();

//     // ---------- int64 safe send ----------
//     int64_t offset = 0;
//     while (offset < numel) {
//         int chunk = static_cast<int>(
//             std::min<int64_t>(INT_MAX, numel - offset));

//         void* chunk_ptr = static_cast<char*>(buffer->data) + offset * type_size;

//         MPI_CALL(MPI_Send(
//             chunk_ptr,
//             chunk,
//             mpi_dtype,
//             /*dest=*/receiver_id,
//             /*tag=*/0,
//             MPI_COMM_WORLD));

//         offset += chunk;
//     }
 
// }

// void RecvFromWorker(Tensor buffer, int sender_id) {
//   CCLThreadLocalContext* ctx = CCLThreadLocalContext::Get();
//   int world_size = ctx->worker->num_workers;
//   int worker_id = ctx->worker->worker_id;
//   CHECK(sender_id >= 0 && sender_id < ctx->worker->num_workers)
//       << "Invalid sender id " << sender_id << ". The world size is " << ctx->worker->num_workers;
//   CHECK_NE(worker_id, sender_id) << "Cannot receive from the worker itself.";
//    DataType dtype(buffer->dtype);
//     MPI_Datatype mpi_dtype = AsMPIDataType(dtype);

//     int type_size;
//     MPI_Type_size(mpi_dtype, &type_size);

//     int64_t numel = buffer.Shape().Product();

//     // ---------- int64 safe recv ----------
//     int64_t offset = 0;
//     while (offset < numel) {
//         int chunk = static_cast<int>(
//             std::min<int64_t>(INT_MAX, numel - offset));

//         void* chunk_ptr = static_cast<char*>(buffer->data) + offset * type_size;

//         MPI_CALL(MPI_Recv(
//             chunk_ptr,
//             chunk,
//             mpi_dtype,
//             /*source=*/sender_id,
//             /*tag=*/0,
//             MPI_COMM_WORLD,
//             MPI_STATUS_IGNORE));

//         offset += chunk;
//     }
  
// }

void SyncWorker() {
  CCLThreadLocalContext* ctx = CCLThreadLocalContext::Get();
  ICHECK(ctx->worker != nullptr);
  MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
}

   TVM_FFI_STATIC_INIT_BLOCK() {
   namespace refl = tvm::ffi::reflection;
   refl::GlobalDef()
       .def("runtime.disco." TVM_DISCO_CCL_NAME ".init_ccl", InitCCL)
       .def("runtime.disco." TVM_DISCO_CCL_NAME ".init_ccl_per_worker", InitCCLPerWorker)
      // .def("runtime.disco." TVM_DISCO_CCL_NAME ".allreduce",
      //      [](Tensor send, int kind, bool in_group, Tensor recv) {
      //        CHECK(0 <= kind && kind <= 4) << "ValueError: Unknown ReduceKind: " << kind;
      //        nccl::AllReduce(send, static_cast<ReduceKind>(kind), in_group, recv);
      //      })
      .def("runtime.disco." TVM_DISCO_CCL_NAME ".allgather",
           [](Tensor send, bool in_group, Tensor recv) { mpi::AllGather(send, in_group, recv); })
      .def("runtime.disco." TVM_DISCO_CCL_NAME ".broadcast_from_worker0", BroadcastFromWorker0)
      .def("runtime.disco." TVM_DISCO_CCL_NAME ".scatter_from_worker0", ScatterFromWorker0)
      .def("runtime.disco." TVM_DISCO_CCL_NAME ".gather_to_worker0", GatherToWorker0)
      .def("runtime.disco." TVM_DISCO_CCL_NAME ".recv_from_worker0", RecvFromWorker0)
      .def("runtime.disco." TVM_DISCO_CCL_NAME ".send_to_next_group", SendToNextGroup)
      .def("runtime.disco." TVM_DISCO_CCL_NAME ".recv_from_prev_group", RecvFromPrevGroup)
      .def("runtime.disco." TVM_DISCO_CCL_NAME ".send_to_worker", SendToWorker)
      .def("runtime.disco." TVM_DISCO_CCL_NAME ".recv_from_worker", RecvFromWorker)
      .def("runtime.disco." TVM_DISCO_CCL_NAME ".sync_worker", SyncWorker);
      // .def("runtime.disco." TVM_DISCO_CCL_NAME ".test_send_to_next_group_recv_from_prev_group",
      //      [](Tensor buffer) {
      //        CCLThreadLocalContext* ctx = CCLThreadLocalContext::Get();
      //        CHECK_EQ(ctx->worker->num_workers, 4) << "The test requires the world size to be 4.";
      //        CHECK_EQ(ctx->worker->num_groups, 2) << "The test requires the group size to be 2.";
      //        int group_size = ctx->worker->num_workers / ctx->worker->num_groups;
      //        int group_id = ctx->worker->worker_id / group_size;
      //        if (group_id == 0) {
      //          tvm::runtime::mpi::SendToNextGroup(buffer);
      //        } else {
      //          tvm::runtime::mpi::RecvFromPrevGroup(buffer);
      //        }
      //      })
      // .def("runtime.disco." TVM_DISCO_CCL_NAME ".test_worker2_sends_to_worker0", [](Tensor buffer) {
      //   CCLThreadLocalContext* ctx = CCLThreadLocalContext::Get();
      //   CHECK_EQ(ctx->worker->num_workers, 4) << "The test requires the world size to be 4.";
      //   CHECK_EQ(ctx->worker->num_groups, 2) << "The test requires the group size to be 2.";
      //   if (ctx->worker->worker_id == 2) {
      //     tvm::runtime::mpi::SendToWorker(buffer, 0);
      //   } else if (ctx->worker->worker_id == 0) {
      //     tvm::runtime::mpi::RecvFromWorker(buffer, 2);
      //   }
      // });

 }

}  // namespace mpi
}  // namespace runtime
}  // namespace tvm
