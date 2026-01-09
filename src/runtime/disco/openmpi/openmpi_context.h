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

#ifndef TVM_RUNTIME_DISCO_OPENMPI_CONTEXT_H_
#define TVM_RUNTIME_DISCO_OPENMPI_CONTEXT_H_

#include <dlpack/dlpack.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/base.h>
#include <tvm/runtime/disco/builtin.h>
#include <tvm/runtime/disco/session.h>

#include <mpi.h>



namespace tvm {
namespace runtime {
namespace openmpi {

#define TVM_DISCO_CCL_NAME "openmpi"
#define TVM_DISCO_DEVICE_NAME "cpu"

const constexpr DLDeviceType TVM_DISCO_DEVICE_TYPE = DLDeviceType::kDLCPU;


#define OPENMPI_CALL(cmd)                                                      
  do {                                                                      
    auto r = (cmd);                                                         
    if (r != MPI_SUCCESS) {        
      char err_string[MPI_MAX_ERROR_STRING];
      LOG(FATAL) << TVM_DISCO_CCL_NAME "Errror: " << MPI_Error_string(r, err_string, nullptr); 
    }                                                                       
  } while (0)


struct CCLThreadLocalContext {
  DiscoWorker* worker = nullptr;
  int device_id;

  ~CCLThreadLocalContext() { Clear(); }

  void Clear() {
    if (group_comm != MPI_COMM_NULL) {
      OPENMPI_CALL(MPI_Comm_free(group_comm));
      if (global_comm == group_comm) {
        global_comm = MPI_COMM_NULL;
      }
      group_comm = MPI_COMM_NULL;
    }
    if (global_comm != MPI_COMM_NULL) {
      OPENMPI_CALL(MPI_Comm_free(global_comm));
      global_comm = MPI_COMM_NULL;
    }
  
    worker = nullptr;
  }

 

  static CCLThreadLocalContext* Get();
};

}  // namespace openmpi
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_DISCO_OPENMPI_CONTEXT_H_
