# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# Variables used by this module, they can change the default behaviour and need
# to be set before calling find_package:


macro(find_openmpi use_openmpi)
  if("${use_openmpi}" MATCHES "${IS_FALSE_PATTERN}")
    return()
  endif()

  if("${use_openmpi}" MATCHES "${IS_TRUE_PATTERN}")
    find_package(MPI)

    if (NOT TARGET MPI::MPI_CXX)
      message(FATAL_ERROR "MPI::MPI_CXX target not available")
    endif()

    message(STATUS "MPI found")
    message(STATUS "MPI C compiler   : ${MPI_C_COMPILER}")
    message(STATUS "MPI CXX compiler : ${MPI_CXX_COMPILER}")
    find_path(OPENMPI_INCLUDE_DIR NAMES mpi.h)

    # 包一層 interface target，與 nccl 用法一致
    add_library(mpi INTERFACE)
    target_link_libraries(mpi INTERFACE MPI::MPI_CXX)

  else()
    message(FATAL_ERROR
      "use_openmpi must be ON/OFF, got: ${use_openmpi}"
    )
  endif()
endmacro()
