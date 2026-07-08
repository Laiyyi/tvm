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

// #ifndef TVM_RUNTIME_DISCO_CPU_CPUCCL_CONTEXT_H_
// #define TVM_RUNTIME_DISCO_CPU_CPUCCL_CONTEXT_H_

// #include <dmlc/io.h>
// #include <tvm/runtime/base.h>
// #include <tvm/runtime/disco/builtin.h>
// #include <tvm/runtime/disco/disco_worker.h>


// namespace tvm {
// namespace runtime {
// namespace cpuccl {


// #ifndef TVM_DISCO_DEVICE_NAME
// #define TVM_DISCO_DEVICE_NAME "cpu"
// #endif
// #ifndef TVM_DISCO_CCL_NAME
// #define TVM_DISCO_CCL_NAME    "cpuccl"
// #endif


// struct CPURingContext {
//   DiscoWorker* worker = nullptr;
//   dmlc::Stream* send_stream = nullptr;  // → right neighbor
//   dmlc::Stream* recv_stream = nullptr;  // ← left  neighbor
//   bool initialized = false;
//   ~CPURingContext() { Clear(); }

//   void Clear() {
//     send_stream = nullptr;
//     recv_stream = nullptr;
//     worker      = nullptr;
//     initialized = false;
//   }

//   static CPURingContext* Get();
// };

// }  // namespace cpuccl
// }  // namespace runtime
// }  // namespace tvm

// #endif  // TVM_RUNTIME_DISCO_CPU_CPUCCL_CONTEXT_H_
