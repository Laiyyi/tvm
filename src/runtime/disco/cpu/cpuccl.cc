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

/*!
 * CPU ring-based collective communication for Disco.
 *
 * Ring topology and connections are established by the session at construction
 * time (when build_ring=true). Each DiscoWorker has:
 *   - ring_out: DiscoChannel* writes to right neighbor
 *   - ring_in:  DiscoChannel* reads from left neighbor
 *
 * This file is pure algorithm — no sockets, no fds, no connection setup.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/disco/builtin.h>
#include <tvm/runtime/disco/disco_worker.h>
#include <tvm/runtime/disco/session.h>


#include <algorithm>
#include <cstring>
#include <vector>

#include "../utils.h"


namespace tvm {
namespace runtime {
namespace cpuccl {

  #ifndef TVM_DISCO_DEVICE_NAME
#define TVM_DISCO_DEVICE_NAME "cpu"
#endif
#ifndef TVM_DISCO_CCL_NAME
#define TVM_DISCO_CCL_NAME    "cpuccl"
#endif

// ─────────────────────────────────────────────────────────────
//  Element-wise reduce helpers
// ─────────────────────────────────────────────────────────────
template <typename T>
static void ReduceInPlace(T* dst, const T* src, int64_t n, ReduceKind op) {
  switch (op) {
    case ReduceKind::kSum:
      for (int64_t i = 0; i < n; ++i) dst[i] += src[i]; break;
    case ReduceKind::kMax:
      for (int64_t i = 0; i < n; ++i) dst[i] = std::max(dst[i], src[i]); break;
    case ReduceKind::kMin:
      for (int64_t i = 0; i < n; ++i) dst[i] = std::min(dst[i], src[i]); break;
    case ReduceKind::kProd:
      for (int64_t i = 0; i < n; ++i) dst[i] *= src[i]; break;
    default:
      LOG(FATAL) << "CPU collective: unsupported ReduceKind " << static_cast<int>(op);
  }
}

static void ReduceBytes(void* dst, const void* src, int64_t numel,
                        DataType dtype, ReduceKind op) {
  if (dtype == DataType::Float(32))
    ReduceInPlace(static_cast<float*>(dst),    static_cast<const float*>(src),    numel, op);
  else if (dtype == DataType::Float(64))
    ReduceInPlace(static_cast<double*>(dst),   static_cast<const double*>(src),   numel, op);
  else if (dtype == DataType::Int(32))
    ReduceInPlace(static_cast<int32_t*>(dst),  static_cast<const int32_t*>(src),  numel, op);
  else if (dtype == DataType::Int(64))
    ReduceInPlace(static_cast<int64_t*>(dst),  static_cast<const int64_t*>(src),  numel, op);
  else if (dtype == DataType::Float(16) || dtype == DataType::BFloat(16))
    ReduceInPlace(static_cast<uint16_t*>(dst), static_cast<const uint16_t*>(src), numel, op);
  else
    LOG(FATAL) << "CPU collective: unsupported dtype " << dtype;
}


void InitCCL(Session sess, ffi::Shape device_ids) {
  DRef func = sess->GetGlobalFunc("runtime.disco.cpu.init_ccl_per_worker");
  DLOG(INFO) << "Initializing " TVM_DISCO_CCL_NAME " with devices: " << device_ids;
  sess->CallPacked(func, device_ids);
}

void InitCCLPerWorker(ffi::Shape /*device_ids*/) {
  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  ICHECK(worker) << "Must be called on a DiscoWorker thread";
  ICHECK(worker->ring_in && worker->ring_out)
      << "CPU ring not built; create session with build_ring=true";
  worker->ccl = "cpu";   // GetCCLFunc 用這個字串拼出 runtime.disco.cpu.<op>
}

// SyncWorker - CPU 沒 GPU stream，純 thread/process 同步點，no-op
void SyncWorker() {}

// // ─────────────────────────────────────────────────────────────
// //  AllReduce - ring-allreduce (reduce-scatter + allgather)
// // ─────────────────────────────────────────────────────────────
void AllReduce(Tensor send, ReduceKind reduce_kind, bool /*in_group*/, Tensor recv) {
  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  ICHECK(worker->ring_in && worker->ring_out) << "Ring not initialized";

  int        N     = worker->num_workers;
  int        rank  = worker->worker_id;
  DataType   dt    = DataType(send->dtype);
  int64_t    numel = send.Shape().Product();
  int64_t    bytes = numel * dt.bytes();

  std::memcpy(recv->data, send->data, static_cast<size_t>(bytes));
  char* buf = static_cast<char*>(recv->data);

  if (N == 1) return;


  int64_t chunk_elem  = (numel + N - 1) / N;       // 每個 chunk 的元素數（ceil 除）
  int64_t chunk_bytes = chunk_elem * dt.bytes();   // 每個 chunk 的 bytes
  std::vector<char> tmp(static_cast<size_t>(chunk_bytes));   // 接收暫存


  // Phase A: reduce-scatter
  for (int r = 0; r < N - 1; ++r) {
    int si = ((rank - r)     % N + N) % N;
    int ri = ((rank - r - 1) % N + N) % N;
    int64_t s_off  = si * chunk_bytes;
    int64_t r_off  = ri * chunk_bytes;
    int64_t s_size = std::min(chunk_bytes, bytes - s_off);
    int64_t r_size = std::min(chunk_bytes, bytes - r_off);
    if (s_size <= 0 || r_size <= 0) continue;

    worker->ring_out->Send(buf + s_off, static_cast<size_t>(s_size));
    worker->ring_in ->Recv(tmp.data(),  static_cast<size_t>(r_size));
    ReduceBytes(buf + r_off, tmp.data(), r_size / dt.bytes(), dt, reduce_kind);
  }

  // Phase B: allgather
  for (int r = 0; r < N - 1; ++r) {
    int si = ((rank + 1 - r) % N + N) % N;
    int ri = ((rank - r)     % N + N) % N;
    int64_t s_off  = si * chunk_bytes;
    int64_t r_off  = ri * chunk_bytes;
    int64_t s_size = std::min(chunk_bytes, bytes - s_off);
    int64_t r_size = std::min(chunk_bytes, bytes - r_off);
    if (s_size <= 0 || r_size <= 0) continue;

    worker->ring_out->Send(buf + s_off, static_cast<size_t>(s_size));
    worker->ring_in ->Recv(buf + r_off, static_cast<size_t>(r_size));
  }
}

// ─────────────────────────────────────────────────────────────
//  AllGather - ring allgather (in-place on recv)
//
//  每個 worker 持有 send (shape=chunk)；輸出 recv shape = (N, chunk)。
//  Worker i 把自己的 chunk 寫進 recv[i]，然後沿 ring 環繞 N-1 輪，
//  每輪送出已知的 chunk、收進下一個未知的 chunk，最終所有 worker
//  都持有完整的 N 個 chunk。
// ─────────────────────────────────────────────────────────────
void AllGather(Tensor send, bool /*in_group*/, Tensor recv) {
  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  ICHECK(worker->ring_in && worker->ring_out);

  int     N           = worker->num_workers;
  int     rank        = worker->worker_id;
  int64_t chunk_bytes = send.Shape().Product() * DataType(send->dtype).bytes();
  char*   buf         = static_cast<char*>(recv->data);

  // 把自己的 chunk 放進 recv 的對應位置 (= recv[rank])
  std::memcpy(buf + rank * chunk_bytes,
              send->data,
              static_cast<size_t>(chunk_bytes));

  // N == 1 沒 ring，直接結束
  if (N == 1) return;

  // 沿 ring 環行 N-1 輪
  // 第 r 輪：送 chunk[(rank - r) % N]，收 chunk[(rank - r - 1) % N]
  for (int r = 0; r < N - 1; ++r) {
    int si = ((rank - r)     % N + N) % N;     // 我這輪要送出去的 chunk index
    int ri = ((rank - r - 1) % N + N) % N;     // 我這輪要收進來的 chunk index

    worker->ring_out->Send(buf + si * chunk_bytes,
                            static_cast<size_t>(chunk_bytes));
    worker->ring_in ->Recv(buf + ri * chunk_bytes,
                            static_cast<size_t>(chunk_bytes));
  }
}


void BroadcastFromWorker0(ffi::Optional<Tensor> send, bool /*in_group*/, Tensor recv) {
  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  ICHECK(worker->ring_in && worker->ring_out);

  int     rank  = worker->worker_id;
  int     N     = worker->num_workers;
  int64_t bytes = recv.Shape().Product() * DataType(recv->dtype).bytes();
LOG(INFO) << "[ccl bcast] wid=" << rank << " N=" << N << " bytes=" << bytes << " ENTER";
  if (rank == 0) {
    ICHECK(send.defined());
    std::memcpy(recv->data, send.value()->data, static_cast<size_t>(bytes));
    if (N > 1) {
  
      worker->ring_out->Send(recv->data, static_cast<size_t>(bytes));  // 傳整塊 recv

    }
  } else {

    worker->ring_in->Recv(recv->data, static_cast<size_t>(bytes));     // 收進 recv

    if (rank < N - 1) {

      worker->ring_out->Send(recv->data, static_cast<size_t>(bytes));  // 轉發給右鄰
  
    }
  }
  
}

void ScatterFromWorker0(ffi::Optional<Tensor> send, bool /*in_group*/, Tensor recv) {
  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  ICHECK(worker->ring_in && worker->ring_out);

  int     rank       = worker->worker_id;
  int     N          = worker->num_workers;
  int64_t recv_bytes = recv.Shape().Product() * DataType(recv->dtype).bytes();

  // N == 1 特例：直接 memcpy
  if (N == 1) {
    ICHECK(send.defined());
    std::memcpy(recv->data, send.value()->data, static_cast<size_t>(recv_bytes));
    std::cerr << "[W" << rank << "] recv=[" << static_cast<float*>(recv->data)[0] << "," << static_cast<float*>(recv->data)[1] << "]" << std::endl;

    return;
  }

  if (rank == 0) {
    // W0：拿完整 send，留 slice 0，送其餘 (N-1) 個 slice
    ICHECK(send.defined()) << "Worker 0 must provide send buffer for scatter";
    char* src = static_cast<char*>(send.value()->data);
    std::memcpy(recv->data, src, static_cast<size_t>(recv_bytes));   // 留自己的 slice 0
    int64_t tail = recv_bytes * (N - 1);                              // 後面 (N-1) 個 slice
    worker->ring_out->Send(src + recv_bytes, static_cast<size_t>(tail));
  } else {
    // Wi：收 (N - rank) 個 slice（= 自己這個 + 比自己右邊的）
    int64_t incoming = recv_bytes * (N - rank);
    std::vector<char> buf(static_cast<size_t>(incoming));
    worker->ring_in->Recv(buf.data(), static_cast<size_t>(incoming));

    // 留 slice 0 = 整體的 slice rank
    std::memcpy(recv->data, buf.data(), static_cast<size_t>(recv_bytes));

    // 若不是鏈尾，把剩下 (N - rank - 1) 個 slice 轉發給右鄰
    if (rank < N - 1) {
      int64_t forward_bytes = incoming - recv_bytes;
      worker->ring_out->Send(buf.data() + recv_bytes,
                              static_cast<size_t>(forward_bytes));
    }
  }
}


TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.disco.cpu.init_ccl",                InitCCL)
      .def("runtime.disco.cpu.init_ccl_per_worker",     InitCCLPerWorker)
      .def("runtime.disco.cpu.allreduce",               AllReduce)
      .def("runtime.disco.cpu.allgather",               AllGather)
      .def("runtime.disco.cpu.broadcast_from_worker0",  BroadcastFromWorker0)
      .def("runtime.disco.cpu.scatter_from_worker0",    ScatterFromWorker0)
      .def("runtime.disco.cpu.sync_worker",             SyncWorker);
}

}  // namespace cpuccl
}  // namespace runtime
}  // namespace tvm

