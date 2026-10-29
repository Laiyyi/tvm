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

import math
from collections import Counter

import tvm
from tvm import IRModule, relax, te, tirx
from tvm.relax.analysis import free_vars
from tvm.relax.expr_functor import PyExprMutator, mutator

_ALLREDUCE = "runtime.disco.allreduce"


def _is_allreduce(value):
    return (
        isinstance(value, relax.Call)
        and value.op.same_as(tvm.ir.Op.get("relax.call_dps_packed"))
        and isinstance(value.args[0], relax.ExternFunc)
        and value.args[0].global_symbol == _ALLREDUCE
    )


def _static_shape(ty):
    return [int(dim) for dim in ty.shape.values]


def _dtype(ty):
    return str(ty.dtype.dtype)


def _nt_matmul(value, out_ty, num_workers):
    if not (isinstance(value, relax.Call) and value.op.same_as(tvm.ir.Op.get("relax.call_tir"))):
        return None
    if not value.args[0].name_hint.startswith("NT_matmul"):
        return None
    args = value.args[1].fields
    if len(args) != 2:
        return None
    a, weight = args
    a_shape, w_shape, out_shape = _static_shape(a.ty), _static_shape(weight.ty), _static_shape(out_ty)
    if len(a_shape) != 2 or a_shape[0] != 1 or len(w_shape) != 2 or w_shape[1] != a_shape[1]:
        return None
    if out_shape != [1, w_shape[0]] or w_shape[0] % num_workers != 0:
        return None
    return a, weight


def _packed(name, *args):
    return relax.Call(relax.ExternFunc(name), list(args), ty_args=[relax.ObjectType()])


def _rotation(rank, start, num_workers):
    order = relax.op.arange(
        relax.prim_value(rank + start), relax.prim_value(rank + start + num_workers), dtype="int64"
    )
    return relax.op.floor_mod(order, relax.const(num_workers, "int64"))


def _tile_func(k_dim, h_dim, width, dtype):
    a = te.placeholder((1, k_dim), dtype, "a")
    weight = te.placeholder((h_dim, k_dim), dtype, "weight")
    offset = tirx.Var("offset", "int64")
    k = te.reduce_axis((0, k_dim), "k")
    out = te.compute(
        (1, width),
        lambda i, j: te.sum(a[i, k] * weight[offset + j, k], axis=k),
        name="NT_matmul_tile",
    )
    return te.create_prim_func([a, weight, offset, out]).without_attr("global_symbol")


@mutator
class _AllReduceDecomposer(PyExprMutator):  # pylint: disable=abstract-method
    def __init__(self, mod, num_workers):
        super().__init__(mod)
        self.num_workers = num_workers
        self.tile_funcs = {}
        self.reset({}, set())

    def reset(self, tiled, skipped):
        self.tiled = tiled
        self.skipped = skipped
        self.rank = None

    def _get_rank(self):
        if self.rank is None:
            worker_id = self.builder_.emit(
                relax.op.call_pure_packed("runtime.disco.worker_id", ty_args=relax.ShapeType(ndim=1)),
                "worker_id",
            )
            self.rank = tirx.Var("rank", "int64")
            self.builder_.match_cast(worker_id, relax.ShapeType([self.rank]))
        return self.rank

    def _alloc(self, chunk, dtype):
        return self.builder_.emit(relax.op.builtin.alloc_tensor(relax.ShapeExpr([1, chunk]), dtype, 0))

    def visit_var_binding_(self, binding):
        if binding.var in self.skipped:
            return None
        if not _is_allreduce(binding.value):
            return super().visit_var_binding_(binding)
        x, reduce_kind, in_group = binding.value.args[1].fields
        reduce_kind, in_group = self.visit_expr(reduce_kind), self.visit_expr(in_group)
        shape, dtype = _static_shape(binding.var.ty), _dtype(binding.var.ty)
        if binding.var in self.tiled:
            shard = self._tiled_reduce_scatter(*self.tiled[binding.var], dtype, reduce_kind, in_group)
        else:
            shard = self._reduce_scatter(self.visit_expr(x), shape, dtype, reduce_kind, in_group)
        self.set_var_remap(binding.var, self._all_gather(shard, shape, dtype, in_group))
        return None

    def _tiled_reduce_scatter(self, a, weight, dtype, reduce_kind, in_group):
        bb = self.builder_
        n = self.num_workers
        h_dim, k_dim = _static_shape(weight.ty)
        width = h_dim // n
        key = (k_dim, h_dim, width, dtype)
        if key not in self.tile_funcs:
            self.tile_funcs[key] = bb.add_func(_tile_func(*key), "NT_matmul_tile")
        rank = self._get_rank()
        a, weight = self.visit_expr(a), self.visit_expr(weight)

        def tile(k):
            offset = tirx.floormod(rank + (n - 1 - k), n) * width
            return bb.emit(
                relax.call_tir(
                    self.tile_funcs[key],
                    [a, weight, relax.prim_value(offset)],
                    relax.TensorType([1, width], dtype),
                )
            )

        bb.emit(_packed("runtime.disco.rs_begin", tile(0), reduce_kind, in_group))
        shard = None
        for k in range(1, n):
            current = tile(k)
            shard = self._alloc(width, dtype)
            bb.emit(_packed("runtime.disco.rs_step", current, in_group, shard))
        return shard

    def _reduce_scatter(self, x, shape, dtype, reduce_kind, in_group):
        bb = self.builder_
        n = self.num_workers
        numel = math.prod(shape)
        assert numel % n == 0, f"allreduce of {shape} cannot be split over {n} workers"
        chunk = numel // n
        rank = self._get_rank()

        rows = bb.emit(relax.op.reshape(x, [n, chunk]))
        rows = bb.emit(relax.op.take(rows, _rotation(rank, 1, n), axis=0))
        parts = bb.emit(relax.op.split(rows, n, axis=0))
        tiles = [bb.emit(relax.TupleGetItem(parts, n - 2 - k)) for k in range(n - 1)]
        tiles.append(bb.emit(relax.TupleGetItem(parts, n - 1)))

        bb.emit(_packed("runtime.disco.rs_begin", tiles[0], reduce_kind, in_group))
        shard = None
        for k in range(1, n):
            shard = self._alloc(chunk, dtype)
            bb.emit(_packed("runtime.disco.rs_step", tiles[k], in_group, shard))
        return shard

    def _all_gather(self, shard, shape, dtype, in_group):
        bb = self.builder_
        n = self.num_workers
        chunk = math.prod(shape) // n
        rank = self._get_rank()

        bb.emit(_packed("runtime.disco.ag_begin", shard, in_group))
        gathered = []
        for _ in range(1, n):
            out = self._alloc(chunk, dtype)
            bb.emit(_packed("runtime.disco.ag_step", in_group, out))
            gathered.append(out)

        rows = bb.emit(relax.op.concat(list(reversed(gathered)) + [shard], axis=0))
        rows = bb.emit(relax.op.take(rows, _rotation(-rank, 2 * n - 1, n), axis=0))
        return bb.emit(relax.op.reshape(rows, shape))


def _find_tiled(func, num_workers):
    bindings = [b for block in func.body.blocks for b in block.bindings]
    uses = Counter(v for b in bindings for v in free_vars(b.value))
    uses.update(free_vars(func.body.body))
    producer = {b.var: b for b in bindings}
    tiled, skipped = {}, set()
    for binding in bindings:
        if not _is_allreduce(binding.value):
            continue
        x = binding.value.args[1].fields[0]
        if x not in producer or uses[x] != 1:
            continue
        matched = _nt_matmul(producer[x].value, x.ty, num_workers)
        if matched is not None:
            tiled[binding.var] = matched
            skipped.add(x)
    return tiled, skipped


@tvm.transform.module_pass(opt_level=0, name="DecomposeAllReduce")
class DecomposeAllReduce:  # pylint: disable=too-few-public-methods
    def __init__(self, num_workers):
        self.num_workers = num_workers

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        if self.num_workers == 1:
            return mod
        mod = relax.transform.ToNonDataflow()(mod)
        decomposer = _AllReduceDecomposer(mod, self.num_workers)
        for g_var, func in mod.functions_items():
            if not isinstance(func, relax.Function):
                continue
            decomposer.reset(*_find_tiled(func, self.num_workers))
            new_func = decomposer.visit_expr(func)
            if not new_func.same_as(func):
                decomposer.builder_.update_func(
                    g_var,
                    relax.Function(
                        new_func.params, new_func.body, new_func.ret_ty, is_pure=False,
                        attrs=new_func.attrs,
                    ),
                )
        return decomposer.builder_.get()
