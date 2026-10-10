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
 * The TCP log is process-wide instead: ProxyLoop runs in threads that are not
 * DiscoWorkers, so a thread_local log would be invisible to the worker that
 * reads it back.  A node's local worker 0 shares the proxy's process, which is
 * what makes the log reachable at all; the other local workers are separate
 * processes and will read an empty log, one entry per node being what we want.
 */
std::mutex g_link_mutex;
std::vector<CallRecord> g_link_records;

}  // namespace

void Record(int64_t op, int64_t bytes, int64_t nanos) {
  g_records.push_back(CallRecord{op, bytes, nanos});
}

void RecordLink(int64_t op, int64_t bytes, int64_t nanos) {
  std::lock_guard<std::mutex> lock(g_link_mutex);
  g_link_records.push_back(CallRecord{op, bytes, nanos});
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
             flat.reserve(g_link_records.size() * 3);
             for (const CallRecord& record : g_link_records) {
               flat.push_back(record.op);
               flat.push_back(record.bytes);
               flat.push_back(record.nanos);
             }
             return ffi::Shape(flat);
           })
      .def("runtime.disco.ccl_timer.reset", []() {
        g_records.clear();
        std::lock_guard<std::mutex> lock(g_link_mutex);
        g_link_records.clear();
      });
}

}  // namespace ccl_timer
}  // namespace runtime
}  // namespace tvm
