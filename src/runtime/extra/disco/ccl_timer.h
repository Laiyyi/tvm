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
 * \file ccl_timer.h
 * \brief Per-call timing for Disco collectives.
 *
 * A CCL backend records one entry per collective by declaring a ScopedTimer at
 * the point where the payload size is known:
 *
 *     int64_t bytes = numel * dtype_bytes;
 *     ccl_timer::ScopedTimer timer(ccl_timer::kAllReduce, bytes);
 *
 * The timer spans the rest of the enclosing scope, so early returns are still
 * accounted for.  Records are read back from Python through
 * "runtime.disco.ccl_timer.records" as a flat Shape of (op, bytes, nanos)
 * triples in call order, and cleared with "runtime.disco.ccl_timer.reset".
 */
#ifndef TVM_RUNTIME_EXTRA_DISCO_CCL_TIMER_H_
#define TVM_RUNTIME_EXTRA_DISCO_CCL_TIMER_H_

#include <chrono>
#include <cstdint>

namespace tvm {
namespace runtime {
namespace ccl_timer {

enum OpId : int64_t {
  kAllReduce = 0,
  kAllGather = 1,
  kBroadcastFromWorker0 = 2,
  kScatterFromWorker0 = 3,
  kGatherToWorker0 = 4,
};

/*!
 * \brief Direction of a TCP hop, recorded separately from the collectives.
 *
 * A collective's cost mixes computation, intra-node pipes and the network.
 * These two isolate the network: the ring crosses a node boundary through the
 * proxy threads, so timing the socket call there is the transfer itself.
 */
enum LinkOpId : int64_t {
  kLinkSend = 0,
  kLinkRecv = 1,
};

/*! \brief Append one record to the calling worker's log. */
void Record(int64_t op, int64_t bytes, int64_t nanos);

/*!
 * \brief Append one TCP hop to the node's log.
 *
 * Kept apart from Record because the proxy threads that cross the node
 * boundary are not DiscoWorker threads, so the log has to be shared by the
 * whole process rather than per worker.
 */
void RecordLink(int64_t op, int64_t bytes, int64_t nanos);

/*!
 * \brief Times the enclosing scope and records it on destruction.
 *
 * RAII rather than explicit stop() calls, because the collectives return early
 * on the single-worker path.
 */
class ScopedTimer {
 public:
  ScopedTimer(int64_t op, int64_t bytes)
      : op_(op), bytes_(bytes), begin_(std::chrono::steady_clock::now()) {}

  ~ScopedTimer() {
    Record(op_, bytes_,
           std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now() - begin_)
               .count());
  }

  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer& operator=(const ScopedTimer&) = delete;

 private:
  int64_t op_;
  int64_t bytes_;
  std::chrono::steady_clock::time_point begin_;
};

/*!
 * \brief Measures a region whose size is only known once it completes.
 *
 * Socket reads and writes return how many bytes they moved, so the payload
 * cannot be passed up front the way ScopedTimer does it.
 */
class Stopwatch {
 public:
  Stopwatch() : begin_(std::chrono::steady_clock::now()) {}

  int64_t ElapsedNanos() const {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now() - begin_)
        .count();
  }

 private:
  std::chrono::steady_clock::time_point begin_;
};

}  // namespace ccl_timer
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_EXTRA_DISCO_CCL_TIMER_H_
