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

#ifndef TVM_RUNTIME_DISCO_MPI_CONTEXT_H_
#define TVM_RUNTIME_DISCO_MPI_CONTEXT_H_

#include <dlpack/dlpack.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/base.h>
#include <tvm/runtime/disco/builtin.h>
#include <tvm/runtime/disco/session.h>

#include <mpi.h>



namespace tvm {
namespace runtime {
namespace mpi {

#define TVM_DISCO_CCL_NAME "mpi"
#define TVM_DISCO_DEVICE_NAME "cpu"
#define MPI_CALL(cmd)                                                       \
  do {                                                                      \
    auto r = (cmd);                                                         \
    if (r != MPI_SUCCESS) {                                                 \
      char err_string[MPI_MAX_ERROR_STRING];                                \
      LOG(FATAL) << TVM_DISCO_CCL_NAME "Errror: " <<  MPI_Error_string(r, err_string, nullptr); \
    }                                                                       \
  } while (0)

const constexpr DLDeviceType TVM_DISCO_DEVICE_TYPE = DLDeviceType::kDLCPU;

/*! \brief Convert DataType to MPIDataType. */
inline MPI_Datatype AsMPIDataType(runtime::DataType dtype) {
  if (dtype == DataType::Int(8) || dtype == DataType::Int(32)) {
    return MPI_INT;
  }
  if (dtype == DataType::UInt(32)) {
    return MPI_UNSIGNED;
  }
  if (dtype == DataType::Int(64)) {
    return MPI_LONG;
  }
  if (dtype == DataType::UInt(64)) {
    return MPI_UNSIGNED_LONG;
  }

  if (dtype == DataType::Float(32)) {
    return MPI_FLOAT;
  }
  if (dtype == DataType::Float(64)) {
    return MPI_DOUBLE;
  }

  if (dtype == DataType::UInt(8) || dtype == DataType::Float8E4M3FN() ||
      dtype == DataType::Float8E5M2() || dtype == DataType::Float(16) ||dtype == DataType::BFloat(16)) {
    // For float8, float16, and bfloat16 data types, pretend to use MPI_UNSIGNED in MPI.
    // Note: Reductions (Allreduce, etc.) may produce incorrect results or throw errors, 
    // because MPI does not natively support these types.
    return MPI_UNSIGNED;
  }
  LOG(FATAL) << "ValueError: Unsupported data type " << dtype;
  throw;
}


struct CCLThreadLocalContext {
  DiscoWorker* worker = nullptr;
  int device_id;
   ~CCLThreadLocalContext() { Clear(); }
  
  void Clear() { worker = nullptr; MPI_CALL(MPI_Finalize());}
  
  static CCLThreadLocalContext* Get();
};

}  // namespace mpi
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_DISCO_MPI_CONTEXT_H_