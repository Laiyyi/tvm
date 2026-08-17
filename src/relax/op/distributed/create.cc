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

#include "create.h"

#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/relax/attrs/create.h>

#include <vector>

namespace tvm {
namespace relax {
namespace distributed {

Type InferDistTypeFull(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 2) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "Full op should have 2 arguments";
  }
  ffi::Array<distributed::DTensorType> input_dtensor_tys = GetInputDTensorType(call, ctx);
  TVM_FFI_ICHECK(input_dtensor_tys.size() == 1);
  distributed::DTensorType fill_value_ty = input_dtensor_tys[0];
  if (fill_value_ty->tensor_ty->ndim != 0) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Full requires the input fill value to be zero rank Tensor. However, the given one is "
        << call->args[1]->ty;
  }
  const auto* attrs = call->attrs.as<InitAttrs>();
  TVM_FFI_ICHECK(attrs);
  ffi::Optional<PrimType> out_dtype = attrs->dtype.has_value()
                                          ? ffi::Optional<PrimType>(PrimType(attrs->dtype.value()))
                                          : fill_value_ty->tensor_ty->dtype;

  // The shape comes from an argument rather than from a distributed input, so there is no input
  // axis for the output to inherit a sharding from: the output is replicated, and a sharded
  // consumer gets a redistribute inserted for it.
  const DeviceMesh& device_mesh = fill_value_ty->device_mesh;
  ffi::Array<PlacementSpec> placement_specs(
      std::vector<PlacementSpec>(device_mesh->shape.size(), PlacementSpec::Replica()));
  return DTensorType(TensorType(/*shape=*/call->args[0], out_dtype), device_mesh,
                     Placement(placement_specs));
}

TVM_REGISTER_OP("relax.full").set_attr<FInferType>("dist.FInferType", InferDistTypeFull);

}  // namespace distributed
}  // namespace relax
}  // namespace tvm
