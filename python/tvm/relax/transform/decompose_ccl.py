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
from collections import defaultdict

import tvm
from tvm import IRModule, relax, te, tirx
from tvm.relax.analysis import free_vars
from tvm.relax.expr_functor import PyExprMutator, mutator

_ALLREDUCE = "runtime.disco.allreduce"
_OUTPUT = -1


def _is_allreduce(value):
    return (
        isinstance(value, relax.Call)
        and value.op.same_as(tvm.ir.Op.get("relax.call_dps_packed"))
        and isinstance(value.args[0], relax.ExternFunc)
        and value.args[0].global_symbol == _ALLREDUCE
    )


def _call_tir_name(value):
    if isinstance(value, relax.Call) and value.op.same_as(tvm.ir.Op.get("relax.call_tir")):
        return value.args[0].name_hint
    return None


def _static_shape(ty):
    return [int(dim) for dim in ty.shape.values]


def _dtype(ty):
    return str(ty.dtype.dtype)


def _packed(name, *args):
    return relax.Call(relax.ExternFunc(name), list(args), ty_args=[relax.ObjectType()])


def _rotation(rank, start, num_workers):
    order = relax.op.arange(
        relax.prim_value(rank + start), relax.prim_value(rank + start + num_workers), dtype="int64"
    )
    return relax.op.floor_mod(order, relax.const(num_workers, "int64"))


def _private(func):
    return func.without_attr("global_symbol")


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
    return _private(te.create_prim_func([a, weight, offset, out]))


def _residual_shard_func(width, residual_shape, dtype):
    shard = te.placeholder((1, width), dtype, "shard")
    residual = te.placeholder(residual_shape, dtype, "residual")
    offset = tirx.Var("offset", "int64")
    lead = (0,) * (len(residual_shape) - 1)
    hidden = te.compute(
        (1, width), lambda i, j: shard[i, j] + residual[lead + (offset + j,)], name="hidden"
    )
    k = te.reduce_axis((0, width), "k")
    square_sum = te.compute(
        (1, 1), lambda i, t: te.sum(hidden[i, k] * hidden[i, k], axis=k), name="square_sum"
    )
    packed = te.compute(
        (1, width + 1),
        lambda i, j: tirx.if_then_else(
            j < width, hidden[i, tirx.min(j, width - 1)], square_sum[i, 0]
        ),
        name="residual_shard",
    )
    return _private(te.create_prim_func([shard, residual, offset, packed]))


def _chunk_gemm_func(width, h_dim, o_dim, dtype, accumulate):
    acc_in = te.placeholder((1, o_dim), dtype, "acc_in")
    packed = te.placeholder((1, width + 1), dtype, "packed")
    gamma = te.placeholder((h_dim,), dtype, "gamma")
    weight = te.placeholder((o_dim, h_dim), dtype, "weight")
    offset = tirx.Var("offset", "int64")
    k = te.reduce_axis((0, width), "k")
    partial = te.compute(
        (1, o_dim),
        lambda i, o: te.sum(
            packed[i, k] * gamma[offset + k] * weight[o, offset + k], axis=k
        ),
        name="chunk_gemm",
    )
    if not accumulate:
        return _private(te.create_prim_func([packed, gamma, weight, offset, partial]))
    out = te.compute((1, o_dim), lambda i, o: acc_in[i, o] + partial[i, o], name="chunk_gemm_acc")
    return _private(te.create_prim_func([acc_in, packed, gamma, weight, offset, out]))


def _rms_scale_func(num_workers, width, h_dim, o_dim, dtype, eps):
    acc = te.placeholder((1, o_dim), dtype, "acc")
    rows = te.placeholder((num_workers, width + 1), dtype, "rows")
    k = te.reduce_axis((0, num_workers), "k")
    total = te.compute((1,), lambda i: te.sum(rows[k, width], axis=k), name="square_total")
    out = te.compute(
        (1, o_dim),
        lambda i, o: acc[i, o]
        * tirx.rsqrt(total[0] / tirx.const(h_dim, dtype) + tirx.const(eps, dtype)),
        name="rms_scale",
    )
    return _private(te.create_prim_func([acc, rows, out]))


def _nt_matmul(value, out_ty, num_workers):
    name = _call_tir_name(value)
    if name is None or not name.startswith("NT_matmul"):
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


def _match_norm_gemm(ar_var, bindings, users, h_dim):
    def only_user(var):
        found = users.get(var, [])
        return bindings[found[0]] if len(found) == 1 and found[0] != _OUTPUT else None

    add = only_user(ar_var)
    if add is None or "add" not in (_call_tir_name(add.value) or ""):
        return None
    add_args = add.value.args[1].fields
    if len(add_args) != 2 or not add_args[0].same_as(ar_var):
        return None
    residual, hidden = add_args[1], add.var
    if math.prod(_static_shape(residual.ty)) != h_dim or math.prod(_static_shape(hidden.ty)) != h_dim:
        return None

    hidden_users = [bindings[i] for i in users.get(hidden, []) if i != _OUTPUT]
    power = next((b for b in hidden_users if (_call_tir_name(b.value) or "").startswith("power")), None)
    if power is None:
        return None
    norm = only_user(power.var)
    if norm is None or "rsqrt" not in (_call_tir_name(norm.value) or ""):
        return None
    norm_args = norm.value.args[1].fields
    if len(norm_args) != 3 or not norm_args[0].same_as(power.var) or not norm_args[1].same_as(hidden):
        return None
    gamma = norm_args[2]
    if _static_shape(gamma.ty) != [h_dim]:
        return None

    gemms = []
    for index in users.get(norm.var, []):
        if index == _OUTPUT:
            return None
        reshape = bindings[index]
        if not (_call_tir_name(reshape.value) or "").startswith("reshape"):
            return None
        matmul = only_user(reshape.var)
        if matmul is None or not (_call_tir_name(matmul.value) or "").startswith("NT_matmul"):
            return None
        mm_args = matmul.value.args[1].fields
        if len(mm_args) != 2 or not mm_args[0].same_as(reshape.var):
            return None
        weight = mm_args[1]
        o_dim, w_h = _static_shape(weight.ty)
        if w_h != h_dim or _static_shape(matmul.var.ty) != [1, o_dim]:
            return None
        gemms.append((reshape.var, matmul.var, weight))
    if not gemms:
        return None

    keep_hidden = any(b is not power and b is not norm for b in hidden_users) or _OUTPUT in users.get(
        hidden, []
    )
    skipped = {hidden, power.var, norm.var}
    for reshape_var, matmul_var, _ in gemms:
        skipped.update((reshape_var, matmul_var))
    return {
        "residual": residual,
        "hidden": hidden,
        "keep_hidden": keep_hidden,
        "gamma": gamma,
        "gemms": [(matmul_var, weight) for _, matmul_var, weight in gemms],
    }, skipped


def _analyze(func, num_workers, rms_norm_eps):
    bindings = [b for block in func.body.blocks for b in block.bindings]
    users = defaultdict(list)
    for index, binding in enumerate(bindings):
        for var in free_vars(binding.value):
            users[var].append(index)
    for var in free_vars(func.body.body):
        users[var].append(_OUTPUT)
    producer = {b.var: b for b in bindings}

    tiled, fused, skipped = {}, {}, set()
    for binding in bindings:
        if not _is_allreduce(binding.value):
            continue
        x = binding.value.args[1].fields[0]
        if x in producer and len(users[x]) == 1:
            matched = _nt_matmul(producer[x].value, x.ty, num_workers)
            if matched is not None:
                tiled[binding.var] = matched
                skipped.add(x)
        shape = _static_shape(binding.var.ty)
        if rms_norm_eps is None or len(shape) != 2 or shape[0] != 1 or shape[1] % num_workers:
            continue
        matched = _match_norm_gemm(binding.var, bindings, users, shape[1])
        if matched is not None:
            fused[binding.var], extra = matched
            skipped.update(extra)
    return tiled, fused, skipped


@mutator
class _AllReduceDecomposer(PyExprMutator):  # pylint: disable=abstract-method
    def __init__(self, mod, num_workers, rms_norm_eps):
        super().__init__(mod)
        self.num_workers = num_workers
        self.rms_norm_eps = rms_norm_eps
        self.funcs = {}
        self.reset({}, {}, set())

    def reset(self, tiled, fused, skipped):
        self.tiled = tiled
        self.fused = fused
        self.skipped = skipped
        self.rank = None

    def _func(self, name, make, *key):
        if (name, key) not in self.funcs:
            self.funcs[(name, key)] = self.builder_.add_func(make(*key), name)
        return self.funcs[(name, key)]

    def _get_rank(self):
        if self.rank is None:
            worker_id = self.builder_.emit(
                relax.op.call_pure_packed("runtime.disco.worker_id", ty_args=relax.ShapeType(ndim=1)),
                "worker_id",
            )
            self.rank = tirx.Var("rank", "int64")
            self.builder_.match_cast(worker_id, relax.ShapeType([self.rank]))
        return self.rank

    def _chunk_offset(self, back, width):
        return tirx.floormod(self._get_rank() + (self.num_workers - back), self.num_workers) * width

    def _alloc(self, size, dtype):
        return self.builder_.emit(relax.op.builtin.alloc_tensor(relax.ShapeExpr([1, size]), dtype, 0))

    def _call(self, gvar, args, shape, dtype):
        return self.builder_.emit(relax.call_tir(gvar, args, relax.TensorType(shape, dtype)))

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
        if binding.var in self.fused:
            self._fused_all_gather(shard, shape[1], dtype, in_group, **self.fused[binding.var])
        else:
            self.set_var_remap(binding.var, self._all_gather(shard, shape, dtype, in_group))
        return None

    def _tiled_reduce_scatter(self, a, weight, dtype, reduce_kind, in_group):
        bb = self.builder_
        n = self.num_workers
        h_dim, k_dim = _static_shape(weight.ty)
        width = h_dim // n
        gvar = self._func("NT_matmul_tile", _tile_func, k_dim, h_dim, width, dtype)
        a, weight = self.visit_expr(a), self.visit_expr(weight)

        def tile(k):
            offset = self._chunk_offset(k + 1, width)
            return self._call(gvar, [a, weight, relax.prim_value(offset)], [1, width], dtype)

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

    def _gather_steps(self, first, size, dtype, in_group):
        bb = self.builder_
        bb.emit(_packed("runtime.disco.ag_begin", first, in_group))
        gathered = []
        for _ in range(1, self.num_workers):
            out = self._alloc(size, dtype)
            bb.emit(_packed("runtime.disco.ag_step", in_group, out))
            gathered.append(out)
        return gathered

    def _restore_order(self, rows):
        n = self.num_workers
        return self.builder_.emit(
            relax.op.take(rows, _rotation(-self._get_rank(), 2 * n - 1, n), axis=0)
        )

    def _all_gather(self, shard, shape, dtype, in_group):
        bb = self.builder_
        chunk = math.prod(shape) // self.num_workers
        gathered = self._gather_steps(shard, chunk, dtype, in_group)
        rows = bb.emit(relax.op.concat(list(reversed(gathered)) + [shard], axis=0))
        return bb.emit(relax.op.reshape(self._restore_order(rows), shape))

    def _fused_all_gather(
        self, shard, h_dim, dtype, in_group, residual, hidden, keep_hidden, gamma, gemms
    ):
        bb = self.builder_
        n = self.num_workers
        width = h_dim // n
        residual, gamma = self.visit_expr(residual), self.visit_expr(gamma)
        residual_shape = _static_shape(residual.ty)
        weights = [self.visit_expr(weight) for _, weight in gemms]
        o_dims = [_static_shape(weight.ty)[0] for weight in weights]

        shard_func = self._func(
            "residual_shard", _residual_shard_func, width, tuple(residual_shape), dtype
        )
        own = self._call(
            shard_func,
            [shard, residual, relax.prim_value(self._chunk_offset(0, width))],
            [1, width + 1],
            dtype,
        )

        def chunk_gemm(packed, back, accs):
            offset = relax.prim_value(self._chunk_offset(back, width))
            result = []
            for index, (weight, o_dim) in enumerate(zip(weights, o_dims)):
                if accs is None:
                    gvar = self._func("chunk_gemm", _chunk_gemm_func, width, h_dim, o_dim, dtype, False)
                    args = [packed, gamma, weight, offset]
                else:
                    gvar = self._func(
                        "chunk_gemm_acc", _chunk_gemm_func, width, h_dim, o_dim, dtype, True
                    )
                    args = [accs[index], packed, gamma, weight, offset]
                result.append(self._call(gvar, args, [1, o_dim], dtype))
            return result

        bb.emit(_packed("runtime.disco.ag_begin", own, in_group))
        accs = chunk_gemm(own, 0, None)
        gathered = []
        for k in range(1, n):
            out = self._alloc(width + 1, dtype)
            bb.emit(_packed("runtime.disco.ag_step", in_group, out))
            gathered.append(out)
            accs = chunk_gemm(out, k, accs)

        rows = bb.emit(relax.op.concat(list(reversed(gathered)) + [own], axis=0))
        for (matmul_var, _), acc, o_dim in zip(gemms, accs, o_dims):
            scale_func = self._func(
                "rms_scale", _rms_scale_func, n, width, h_dim, o_dim, dtype, self.rms_norm_eps
            )
            self.set_var_remap(matmul_var, self._call(scale_func, [acc, rows], [1, o_dim], dtype))

        if keep_hidden:
            ordered = self._restore_order(rows)
            values = bb.emit(relax.op.strided_slice(ordered, axes=[1], begin=[0], end=[width]))
            self.set_var_remap(
                hidden, bb.emit(relax.op.reshape(values, _static_shape(hidden.ty)))
            )


@tvm.transform.module_pass(opt_level=0, name="DecomposeAllReduce")
class DecomposeAllReduce:  # pylint: disable=too-few-public-methods
    def __init__(self, num_workers, rms_norm_eps=None):
        self.num_workers = num_workers
        self.rms_norm_eps = rms_norm_eps

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        if self.num_workers == 1:
            return mod
        mod = relax.transform.ToNonDataflow()(mod)
        decomposer = _AllReduceDecomposer(mod, self.num_workers, self.rms_norm_eps)
        for g_var, func in mod.functions_items():
            if not isinstance(func, relax.Function):
                continue
            decomposer.reset(*_analyze(func, self.num_workers, self.rms_norm_eps))
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
