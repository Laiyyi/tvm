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

#include "search.h"

#include <tvm/ffi/extra/visit_error_context.h>

namespace tvm {
namespace relax {
namespace distributed {

Type InferDistTypeWhere(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 3) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "Where op should take 3 arguments";
  }
  ffi::Array<distributed::DTensorType> input_dtensor_tys = GetInputDTensorType(call, ctx);
  TVM_FFI_ICHECK(input_dtensor_tys.size() == 3);
  TensorType cond_ty = input_dtensor_tys[0]->tensor_ty;
  TensorType x1_ty = input_dtensor_tys[1]->tensor_ty;
  TensorType x2_ty = input_dtensor_tys[2]->tensor_ty;

  // The condition check only cares about the boolean element kind, as in the non-distributed one.
  if (!cond_ty->IsUnknownDtype() && !cond_ty->dtype.value().MatchesCode(DLDataTypeCode::kDLBool)) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "Where requires the input condition tensor to have boolean dtype. However, "
           "the given condition dtype is "
        << cond_ty->dtype;
  }
  ffi::Optional<PrimType> output_dtype = InferBinaryArithOpOutDtype(call, ctx, x1_ty, x2_ty);

  const auto* cond_shape = cond_ty->shape.as<ShapeExprNode>();
  const auto* x1_shape = x1_ty->shape.as<ShapeExprNode>();
  const auto* x2_shape = x2_ty->shape.as<ShapeExprNode>();
  if (cond_shape == nullptr || x1_shape == nullptr || x2_shape == nullptr) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "Input of distributed operator must have known shape";
  }
  ffi::Optional<ffi::Array<PrimExpr>> output_shape =
      InferBinaryBroadcastShape(call, ctx, x1_shape->values, x2_shape->values);
  if (output_shape.has_value()) {
    output_shape = InferBinaryBroadcastShape(call, ctx, cond_shape->values, output_shape.value());
  }
  if (!output_shape.has_value()) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "Cannot broadcast the inputs of " << call->op;
  }
  TensorType output_tensor_ty(ShapeExpr(output_shape.value()), output_dtype);
  return InferShardingSpec(call, ctx, output_tensor_ty, distributed::BuildAxisGraphWhere);
}

TVM_REGISTER_OP("relax.where").set_attr<FInferType>("dist.FInferType", InferDistTypeWhere);

}  // namespace distributed
}  // namespace relax
}  // namespace tvm
