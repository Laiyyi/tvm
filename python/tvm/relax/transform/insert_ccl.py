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
"""Tensor parallel 用的 pass 把 collective 插進 Relax 圖裡。

row parallel 切的權重，每個 worker 只算出部分和，必須跨 worker 做 reduce
column parallel 切的權重，各 worker 持有結果的不同切片，不需要通訊。

兩種觸發方式：

* 看權重 —— InsertAllReduce / InsertAllGather / InsertGatherToWorker0
  把 matmul 的 RHS 回溯到它的權重參數，再決定要不要在輸出插 collective。
* 看權重，但插在 reshape 之後 —— InsertLogitsAllGather
  前端把 nn.Linear 作用在 3-D 張量上時會先壓掉 batch 維再 reshape 還原，
  而 reshape 的目標形狀是寫死的，所以 lm_head 那種情況不能插在 matmul 之後。
* 看參數 —— InsertBroadcastFromWorker0 / InsertScatterFromWorker0
  替換指定的 function param 讓它在進入計算前先過一道 collective。

全部都必須排在 LegalizeOps 和 FuseTransposeMatmul 之前 —— 那兩個 pass 會把
relax.matmul 改寫成 TIR function之後就找不到 matmul 可以配對了。

改寫一律是「回傳巢狀運算式」，由 block builder 的 normalizer 自己拆成 binding
（py_expr_functor.cc:351 的 builder_->Normalize），所以這裡不需要手動 emit。
注意不能改用 dpl 的 rewrite_call —— 它跑到 fixpoint 才停
（dataflow_expr_rewriter.cc:838），而這些改寫是「包住」配對到的 matmul，
包完 matmul 還在，下一輪又被配到，永遠不收斂。
"""

import tvm
from tvm import IRModule, relax
from tvm.relax.expr_functor import PyExprMutator, mutator


REDUCE_OPS = ("sum", "prod", "min", "max", "avg")

_MATMUL_OP = "relax.matmul"
_PERMUTE_OP = "relax.permute_dims"
_RESHAPE_OP = "relax.reshape"


# --------------------------------------------------------------------------- #
# 共用工具
# --------------------------------------------------------------------------- #


def _static_shape(expr):
    """ Get static shape."""
    ty = getattr(expr, "ty", None)
    if not isinstance(ty, relax.TensorType) or ty.shape is None:
        return None
    try:
        return tuple(int(d) for d in ty.shape.values)
    except (AttributeError, TypeError, ValueError):
        return None


def _resolve_binding(expr, lookup_binding):
    """穿過 dataflow var 的 binding，追到真正的運算式或 function param。"""
    while isinstance(expr, relax.Var):
        bound = lookup_binding(expr)
        if bound is None:
            return expr  # 查不到 binding，就是 function param
        expr = bound
    return expr


def _trace_to_param(expr, lookup_binding, permute_op):
    """把 matmul 的 RHS 回溯到它的權重參數。

    前端把 nn.Linear 展開成 permute_dims(W) 再 matmul，中間還會經過
    dataflow var 的 binding，所以要同時穿過這兩層。

    回傳 (param_or_None, transposed)，transposed 記錄途中做了奇數次
    還是偶數次轉置 —— 這決定 W 的哪個軸變成 matmul 的收縮維。
    """
    transposed = False
    while True:
        if isinstance(expr, relax.Var):
            bound = lookup_binding(expr)
            if bound is None:
                return expr, transposed  # 查不到 binding，就是 function param
            expr = bound
        elif isinstance(expr, relax.Call) and expr.op == permute_op:
            transposed = not transposed
            expr = expr.args[0]
        else:
            return None, transposed


def _concat_along_axis(x, axis, make_collective):
    """沿 axis 把各 worker 的切片拼成完整張量。

    allgather 和 gather_to_worker0 都只把 dim 0 乘上 num_workers
    （ccl.cc:95、ccl.cc:203），runtime 那端也是把各 rank 的 chunk
    依序平鋪進 buffer（cpuccl.cc:204）。所以要沿其他軸拼接，得先把那一軸
    轉到最前面，拼完再轉回來。column parallel 切的是最後一維，必走這條路。
    """
    ndim = x.ty.ndim
    target = axis % ndim
    if target == 0:
        return make_collective(x)

    perm = [target] + [i for i in range(ndim) if i != target]
    inverse = [perm.index(i) for i in range(ndim)]
    return relax.op.permute_dims(
        make_collective(relax.op.permute_dims(x, axes=perm)), axes=inverse
    )


def _rewrite_functions(mod, make_inserter):
    """對 module 裡每個 Relax function 跑一次 inserter，TIR PrimFunc 跳過。

    每個 function 用一個新的 inserter —— 它內部的快取只在自己那個 block 有效。
    """
    updates = {}
    for g_var, func in mod.functions_items():
        if not isinstance(func, relax.Function):
            continue
        new_func = make_inserter(mod).visit_expr(func)
        if not new_func.same_as(func):
            updates[g_var] = new_func

    if updates:
        mod = mod.clone()
        for g_var, func in updates.items():
            mod[g_var] = func
    return mod


# --------------------------------------------------------------------------- #
# all-reduce：row parallel 的部分和
# --------------------------------------------------------------------------- #


@tvm.transform.module_pass(opt_level=0, name="InsertAllReduce")
class InsertAllReduce:  # pylint: disable=too-few-public-methods
    """在 row-parallel 的 matmul 後面插 all-reduce。

    兩種指認 row-parallel 的方式，擇一：

    shard_axis : Optional[Dict[str, int]]
        權重參數名 -> 切的軸 0 = column parallel 1 = row parallel
        pass 會把每個 matmul 的 RHS 回溯到它的權重參數，查這張表決定要不要插。
        這是建議的用法：名字是唯一的，不會誤判。

    row_parallel_rhs : Optional[Iterable[Tuple[int, ...]]]
        改用 matmul RHS 的靜態形狀來配對。沒有 shard_axis 時的退路，
        但任何形狀剛好相同的 matmul 都會被插，要自己確認沒有誤判。

    reduce_op : str
        REDUCE_OPS

    in_group : bool
        True 只在 group 內。num_groups=1 時兩者等價 cpuccl 的實作忽略這個參數
    """

    def __init__(self, shard_axis=None, row_parallel_rhs=None, reduce_op="sum", in_group=True):
        if (shard_axis is None) == (row_parallel_rhs is None):
            raise ValueError("shard_axis 和 row_parallel_rhs 必須給且只給一個")
        if reduce_op not in REDUCE_OPS:
            raise ValueError(f"reduce_op 必須是 {REDUCE_OPS} 之一，收到 {reduce_op!r}")

        self.shard_axis = dict(shard_axis) if shard_axis is not None else None
        self.row_parallel_rhs = (
            {tuple(shape) for shape in row_parallel_rhs} if row_parallel_rhs is not None else None
        )
        self.reduce_op = reduce_op
        self.in_group = in_group

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        def make_inserter(mod):
            return _AllReduceInserter(
                mod, self.shard_axis, self.row_parallel_rhs, self.reduce_op, self.in_group
            )

        return _rewrite_functions(mod, make_inserter)


@mutator
class _AllReduceInserter(PyExprMutator):  # pylint: disable=abstract-method
    def __init__(self, mod, shard_axis, row_parallel_rhs, reduce_op, in_group):
        super().__init__(mod)
        self.shard_axis = shard_axis
        self.row_parallel_rhs = row_parallel_rhs
        self.reduce_op = reduce_op
        self.in_group = in_group
        self.matmul_op = tvm.ir.Op.get(_MATMUL_OP)
        self.permute_op = tvm.ir.Op.get(_PERMUTE_OP)

    def _needs_allreduce(self, rhs):
        if self.shard_axis is not None:
            param, transposed = _trace_to_param(rhs, self.lookup_binding, self.permute_op)
            if param is None:
                return False
            axis = self.shard_axis.get(param.name)
            if axis is None:
                return False
            # 權重 W 的形狀是 (out, in)。matmul 收縮 RHS 的倒數第二維：
            #   transposed（RHS = W^T）→ RHS[-2] = W[1] = in  → 收縮軸是 W 的 axis 1
            #   未轉置（RHS = W）      → RHS[-2] = W[0] = out → 收縮軸是 W 的 axis 0
            # 切的軸正好是收縮軸時，各 worker 只有部分和，才需要 all-reduce。
            return int(axis) == (1 if transposed else 0)

        return _static_shape(rhs) in self.row_parallel_rhs

    def visit_call_(self, call):  # pylint: disable=arguments-renamed
        call = self.visit_expr_post_order(call)  # 先處理子節點，巢狀的 matmul 才不會漏
        if call.op != self.matmul_op:
            return call
        if not self._needs_allreduce(call.args[1]):
            return call
        return relax.op.ccl.allreduce(call, self.reduce_op, self.in_group)


# --------------------------------------------------------------------------- #
# all-gather / gather：把 column parallel 的輸出切片拼回完整張量
# --------------------------------------------------------------------------- #


@tvm.transform.module_pass(opt_level=0, name="InsertAllGather")
class InsertAllGather:  # pylint: disable=too-few-public-methods
    """在指定權重的 matmul 輸出插 all-gather，每個 worker 都拿到完整結果。

    用在 column parallel 的尾端：下游如果不是另一個 row-parallel matmul
    （例如要接 norm、要當 logits 取樣），切片就得先拼回完整張量。
    插了之後輸出形狀會變大 num_workers 倍，下游必須預期完整張量。

    param_names : Iterable[str]
        要拼接的權重參數名。只有 matmul 的 RHS 回溯到這些參數時才插。

    num_workers : int
        拼接的份數，等於 TP。

    axis : int
        輸出的哪一軸要拼。column parallel 切的是輸出維，所以預設 -1。

    in_group : bool
        同 InsertAllReduce。
    """

    def __init__(self, param_names, num_workers, axis=-1, in_group=True):
        self.param_names = set(param_names)
        self.num_workers = int(num_workers)
        self.axis = int(axis)
        self.in_group = in_group

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        def make_collective(x):
            return relax.op.ccl.allgather(x, self.num_workers, self.in_group)

        def make_inserter(mod):
            return _MatmulOutputCollective(mod, self.param_names, self.axis, make_collective)

        return _rewrite_functions(mod, make_inserter)


@tvm.transform.module_pass(opt_level=0, name="InsertGatherToWorker0")
class InsertGatherToWorker0:  # pylint: disable=too-few-public-methods
    """在指定權重的 matmul 輸出插 gather，只有 worker 0 拿到完整結果。

    比 all-gather 省通訊，但其他 worker 上那塊記憶體的內容沒有定義，
    所以只適合用在圖的最末端 —— 結果馬上要被 copy_from_worker_0 搬回 host。

    參數同 InsertAllGather。
    """

    def __init__(self, param_names, num_workers, axis=-1, in_group=True):
        self.param_names = set(param_names)
        self.num_workers = int(num_workers)
        self.axis = int(axis)
        self.in_group = in_group

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        def make_collective(x):
            return relax.op.ccl.gather_to_worker0(x, self.num_workers, self.in_group)

        def make_inserter(mod):
            return _MatmulOutputCollective(mod, self.param_names, self.axis, make_collective)

        return _rewrite_functions(mod, make_inserter)


@mutator
class _MatmulOutputCollective(PyExprMutator):  # pylint: disable=abstract-method
    """matmul 的 RHS 回溯到指定參數時，在輸出插一道沿 axis 拼接的 collective。"""

    def __init__(self, mod, param_names, axis, make_collective):
        super().__init__(mod)
        self.param_names = param_names
        self.axis = axis
        self.make_collective = make_collective
        self.matmul_op = tvm.ir.Op.get(_MATMUL_OP)
        self.permute_op = tvm.ir.Op.get(_PERMUTE_OP)

    def visit_call_(self, call):  # pylint: disable=arguments-renamed
        call = self.visit_expr_post_order(call)
        if call.op != self.matmul_op:
            return call

        param, _ = _trace_to_param(call.args[1], self.lookup_binding, self.permute_op)
        if param is None or param.name not in self.param_names:
            return call

        # visit_expr_post_order 已經 normalize 過，call.ty 有形狀可以算轉置順序
        return _concat_along_axis(call, self.axis, self.make_collective)


@tvm.transform.module_pass(opt_level=0, name="InsertLogitsAllGather")
class InsertLogitsAllGather:  # pylint: disable=too-few-public-methods
    """在指定權重的 matmul「後面那個 reshape」之後插 all-gather。

    跟 InsertAllGather 的差別只有觸發點。前端把 nn.Linear 作用在 3-D 張量上時
    會先把 batch 維壓掉再還原，所以 lm_head 的尾巴長這樣：

        lv1732 = R.matmul(lv1731, lv1730)                      # (1, 16032)
        lv1733 = R.reshape(lv1732, R.shape([1, 1, 16032]))     # ← 插這之後

    reshape 的目標形狀是寫死的，所以 InsertAllGather（以 matmul 為觸發點）插進去
    之後下一行就對不上，會噴 "Reshape expects the new shape to be convertible
    from the old shape"。這個 pass 改成以 reshape 為觸發點，回溯它的輸入確認是
    該權重的 matmul 才插。

    param_names : Iterable[str]
        要拼接的權重參數名。

    num_workers : int
        拼接的份數，等於 TP。

    axis : int
        reshape 輸出的哪一軸要拼。column parallel 切的是輸出維，所以預設 -1。

    in_group : bool
        同 InsertAllReduce。
    """

    def __init__(self, param_names, num_workers, axis=-1, in_group=True):
        self.param_names = set(param_names)
        self.num_workers = int(num_workers)
        self.axis = int(axis)
        self.in_group = in_group

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        def make_collective(x):
            return relax.op.ccl.allgather(x, self.num_workers, self.in_group)

        def make_inserter(mod):
            return _ReshapedMatmulOutputCollective(
                mod, self.param_names, self.axis, make_collective
            )

        return _rewrite_functions(mod, make_inserter)


@mutator
class _ReshapedMatmulOutputCollective(PyExprMutator):  # pylint: disable=abstract-method
    """reshape 的輸入是指定權重的 matmul 時，在 reshape 之後插 collective。"""

    def __init__(self, mod, param_names, axis, make_collective):
        super().__init__(mod)
        self.param_names = param_names
        self.axis = axis
        self.make_collective = make_collective
        self.reshape_op = tvm.ir.Op.get(_RESHAPE_OP)
        self.matmul_op = tvm.ir.Op.get(_MATMUL_OP)
        self.permute_op = tvm.ir.Op.get(_PERMUTE_OP)

    def visit_call_(self, call):  # pylint: disable=arguments-renamed
        call = self.visit_expr_post_order(call)
        if call.op != self.reshape_op:
            return call

        inner = _resolve_binding(call.args[0], self.lookup_binding)
        if not (isinstance(inner, relax.Call) and inner.op == self.matmul_op):
            return call

        param, _ = _trace_to_param(inner.args[1], self.lookup_binding, self.permute_op)
        if param is None or param.name not in self.param_names:
            return call

        # visit_expr_post_order 已經 normalize 過，call.ty 有形狀可以算轉置順序
        return _concat_along_axis(call, self.axis, self.make_collective)


# --------------------------------------------------------------------------- #
# broadcast / scatter：改寫 function param 本身
# --------------------------------------------------------------------------- #


@tvm.transform.module_pass(opt_level=0, name="InsertBroadcastFromWorker0")
class InsertBroadcastFromWorker0:  # pylint: disable=too-few-public-methods
    """讓指定的 function param 先從 worker 0 broadcast 出去再被使用。

    用在「只有 worker 0 手上有正確值」的輸入 —— host 端用 sess.broadcast
    逐個傳會多繞一趟 controller，交給圖內做可以省掉。

    param_names : Iterable[str]
        要 broadcast 的 function param 名稱。
    """

    def __init__(self, param_names):
        self.param_names = set(param_names)

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        table = {name: relax.op.ccl.broadcast_from_worker0 for name in self.param_names}
        return _rewrite_functions(mod, lambda mod: _ParamSubstituter(mod, table))


@tvm.transform.module_pass(opt_level=0, name="InsertScatterFromWorker0")
class InsertScatterFromWorker0:  # pylint: disable=too-few-public-methods
    """讓指定的 function param 先從 worker 0 scatter 出去再被使用。

    用在「完整權重送進圖、由圖自己切」的情境：host 端全部走 broadcast，
    每個 worker 在圖內取走自己那一份，不必在 host 端先 np.split + stack。
    代價是完整張量得先在每個 worker 上存在一次。

    param_axis : Dict[str, int]
        參數名 -> 要切的軸。該軸的長度必須被 num_workers 整除，
        不整除的話 scatter_from_worker0 的型別推導會直接報錯（ccl.cc:152）。
        切完形狀就變了，下游必須預期切過的形狀。

    num_workers : int
        切成幾份，等於 TP。
    """

    def __init__(self, param_axis, num_workers):
        self.param_axis = dict(param_axis)
        self.num_workers = int(num_workers)

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        def make(axis):
            return lambda x: relax.op.ccl.scatter_from_worker0(x, self.num_workers, axis)

        table = {name: make(int(axis)) for name, axis in self.param_axis.items()}
        return _rewrite_functions(mod, lambda mod: _ParamSubstituter(mod, table))


@mutator
class _ParamSubstituter(PyExprMutator):  # pylint: disable=abstract-method
    """把 function param 的每個使用點換成「過了 collective 的那個運算式」。

    同一個 param 被用多次的話會插多道 collective —— 這個模型裡每個權重只餵給
    一個 matmul，所以實際上是一對一。
    """

    def __init__(self, mod, table):
        super().__init__(mod)
        self.table = table

    def visit_var_(self, var):  # pylint: disable=arguments-renamed
        if var.name not in self.table:
            return var
        # 有 binding 的是中間結果，不是 function param，名字撞到也不能改
        if self.lookup_binding(var) is not None:
            return var
        return self.table[var.name](var)
