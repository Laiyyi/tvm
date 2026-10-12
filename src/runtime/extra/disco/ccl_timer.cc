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
 * \file ccl_timer.cc
 * \brief Storage and Python-facing accessors for ccl_timer.h.
 */

#include "./ccl_timer.h"

#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/reflection/registry.h>

#include <limits>
#include <mutex>
#include <vector>

namespace tvm {
namespace runtime {
namespace ccl_timer {

namespace {

struct CallRecord {
  int64_t op;
  int64_t bytes;
  int64_t nanos;
};

/*
 * thread_local keeps one log per worker: a node's local worker 0 runs as a
 * thread inside the proxy process while worker 1 is a separate process, and
 * both end up with their own log either way.
 */
thread_local std::vector<CallRecord> g_records;

/*
 * The proxy stats are process-wide instead: ProxyLoop runs in threads that are
 * not DiscoWorkers, so a thread_local log would be invisible to the worker that
 * reads it back.  A node's local worker 0 shares the proxy's process, which is
 * what makes the stats reachable at all; the other local workers are separate
 * processes and will read zeros, one entry per node being what we want.
 *
 * Aggregated rather than logged per call: a 500-token run drives the proxy
 * ~200k times per direction, and the aggregate answers every question a full
 * log would -- how large each transfer was (size_buckets), whether the socket
 * ever pushed back (partial_writes), and how the time split between waiting
 * for data (read_nanos) and handing it on (write_nanos).
 */
struct LinkStats {
  int64_t iterations;
  int64_t bytes;
  int64_t read_nanos;
  int64_t write_nanos;
  int64_t write_calls;
  int64_t partial_writes;
  int64_t min_bytes;
  int64_t max_bytes;
  int64_t size_buckets[kLinkSizeBuckets];
  /*
   * The first iteration after a reset is dropped: the proxy thread was already
   * blocked inside read() before the reset landed, so its read_nanos covers
   * time from outside the measurement window.
   */
  bool seen_first;
};

constexpr int kLinkStatsWidth = 8 + kLinkSizeBuckets;

std::mutex g_link_mutex;
LinkStats g_link_stats[2];

void ClearLinkStats() {
  for (LinkStats& stats : g_link_stats) {
    stats = LinkStats{};
    stats.min_bytes = std::numeric_limits<int64_t>::max();
  }
}

int SizeBucket(int64_t bytes) {
  int bucket = 0;
  while (bucket + 1 < kLinkSizeBuckets && (int64_t{1} << (bucket + 1)) <= bytes) {
    ++bucket;
  }
  return bucket;
}

}  // namespace

void Record(int64_t op, int64_t bytes, int64_t nanos) {
  g_records.push_back(CallRecord{op, bytes, nanos});
}

void RecordLink(int64_t op, int64_t bytes, int64_t read_nanos, int64_t write_nanos,
                int64_t write_calls) {
  if (op < 0 || op > 1) return;
  std::lock_guard<std::mutex> lock(g_link_mutex);
  LinkStats& stats = g_link_stats[op];
  if (!stats.seen_first) {
    stats.seen_first = true;
    return;
  }
  stats.iterations += 1;
  stats.bytes += bytes;
  stats.read_nanos += read_nanos;
  stats.write_nanos += write_nanos;
  stats.write_calls += write_calls;
  if (write_calls > 1) stats.partial_writes += 1;
  if (stats.iterations == 1 || bytes < stats.min_bytes) stats.min_bytes = bytes;
  if (bytes > stats.max_bytes) stats.max_bytes = bytes;
  stats.size_buckets[SizeBucket(bytes)] += 1;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.disco.ccl_timer.records",
           []() -> ffi::Shape {
             std::vector<int64_t> flat;
             flat.reserve(g_records.size() * 3);
             for (const CallRecord& record : g_records) {
               flat.push_back(record.op);
               flat.push_back(record.bytes);
               flat.push_back(record.nanos);
             }
             return ffi::Shape(flat);
           })
      .def("runtime.disco.ccl_timer.link_records",
           []() -> ffi::Shape {
             std::lock_guard<std::mutex> lock(g_link_mutex);
             std::vector<int64_t> flat;
             flat.reserve(2 * kLinkStatsWidth);
             for (const LinkStats& stats : g_link_stats) {
               flat.push_back(stats.iterations);
               flat.push_back(stats.bytes);
               flat.push_back(stats.read_nanos);
               flat.push_back(stats.write_nanos);
               flat.push_back(stats.write_calls);
               flat.push_back(stats.partial_writes);
               flat.push_back(stats.iterations == 0 ? 0 : stats.min_bytes);
               flat.push_back(stats.max_bytes);
               for (int bucket = 0; bucket < kLinkSizeBuckets; ++bucket) {
                 flat.push_back(stats.size_buckets[bucket]);
               }
             }
             return ffi::Shape(flat);
           })
      .def("runtime.disco.ccl_timer.link_stats_width",
           []() -> int64_t { return kLinkStatsWidth; })
      .def("runtime.disco.ccl_timer.reset", []() {
        g_records.clear();
        std::lock_guard<std::mutex> lock(g_link_mutex);
        ClearLinkStats();
      });
}

}  // namespace ccl_timer
}  // namespace runtime
}  // namespace tvm
