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

#include "index.h"

#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/relax/attrs/index.h>

namespace tvm {
namespace relax {
namespace distributed {

Type InferDistTypeTake(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 2) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "Take op should take 2 arguments";
  }
  ffi::Array<distributed::DTensorType> input_dtensor_tys = GetInputDTensorType(call, ctx);
  TensorType data_ty = input_dtensor_tys[0]->tensor_ty;

  TensorType indices_ty = [&]() {
    Expr arg = call->args[1];
    if (const auto* dtensor_ty = GetTypeAs<distributed::DTensorTypeNode>(arg)) {
      return dtensor_ty->tensor_ty;
    } else if (const auto* prim_ty = GetTypeAs<PrimTypeNode>(arg)) {
      return TensorType(ShapeExpr(ffi::Array<PrimExpr>{}), ffi::GetRef<PrimType>(prim_ty));
    } else {
      TVM_FFI_VISIT_THROW(TypeError, call)
          << "Operator " << call->op << " requires the indices argument to be "
          << "either a tensor or a scalar value. However, argument " << arg << " has type "
          << arg->ty;
      TVM_FFI_UNREACHABLE();
    }
  }();

  const auto* attrs = call->attrs.as<TakeAttrs>();
  TVM_FFI_ICHECK(attrs);
  if (data_ty->IsUnknownNdim() || indices_ty->IsUnknownNdim()) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "Input of distributed operator must have known ndim";
  }
  if (!attrs->axis.has_value() && data_ty->ndim != 1) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Take op expects the input data to be 1-dimensional tensor when the axis "
           "is not specified. However, the given data tensor has ndim "
        << data_ty->ndim;
  }
  int axis = 0;
  if (attrs->axis.has_value()) {
    axis = NormalizeAxis(call, ctx, data_ty->ndim, attrs->axis.value());
  }

  const auto* data_shape = data_ty->shape.as<ShapeExprNode>();
  const auto* indices_shape = indices_ty->shape.as<ShapeExprNode>();
  if (data_shape == nullptr || indices_shape == nullptr) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "Input of distributed operator must have known shape";
  }
  ffi::Array<PrimExpr> out_shape;
  for (int i = 0; i < data_ty->ndim; i++) {
    if (i == axis) {
      for (int j = 0; j < indices_ty->ndim; j++) {
        out_shape.push_back(indices_shape->values[j]);
      }
    } else {
      out_shape.push_back(data_shape->values[i]);
    }
  }
  TensorType output_tensor_ty(ShapeExpr(out_shape), data_ty->dtype);
  return InferShardingSpec(call, ctx, output_tensor_ty, distributed::BuildAxisGraphTake);
}

TVM_REGISTER_OP("relax.take").set_attr<FInferType>("dist.FInferType", InferDistTypeTake);

}  // namespace distributed
}  // namespace relax
}  // namespace tvm
