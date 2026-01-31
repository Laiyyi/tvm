# 測試靠 relax 中有 ccl.scatter(allreduce) 等等會呼叫 disco 的 collective communication 
# 這樣是否還需要直接把資料tensor是否需要靠 mpi 拆分
# 另外這個也是測試直接是否可用relax call mpi4py (這樣就不用做c++)
import numpy as np
import os
# mpiexec --host node1,node2,node3,node4 -n 4 /home/buntu/tvm-env/bin/python3 TestMPI4pyCCL.py
from mpi4py import MPI
import tvm
from tvm import relax

from tvm.script import ir as I
from tvm.script import relax as R

# @I.ir_module
# class Module:
#     @R.function
#     def forward(x: R.Tensor((2, 2), dtype="float32"), fc1_weight: R.Tensor((2, 2), dtype="float32"), fc1_bias: R.Tensor((2, 2), dtype="float32")) -> R.Tensor((2, 1), dtype="float32"):
#         R.func_attr({"num_input": 1})
#         with R.dataflow():
#             lv: R.Tensor((2, 2), dtype="float32") = R.ccl.broadcast_from_worker0(x)
#             lv1: R.Tensor((1, 2), dtype="float32") = R.ccl.scatter_from_worker0(fc1_weight, num_workers=2, axis=0)
#             lv2: R.Tensor((2, 1), dtype="float32") = R.ccl.scatter_from_worker0(fc1_bias, num_workers=2, axis=1)
#             permute_dims: R.Tensor((2, 1), dtype="float32") = R.permute_dims(lv1, axes=None)
#             matmul: R.Tensor((2, 1), dtype="float32") = R.matmul(lv, permute_dims, out_dtype="void")
#             _add: R.Tensor((2, 1), dtype="float32") = R.add(matmul, lv2)
#             gv: R.Tensor((2, 1), dtype="float32") = _add
#             R.output(gv)
#         return gv

# lowerdistir = tvm.relax.transform.LegalizeOps()(Module)

from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def add(matmul: T.Buffer((T.int64(2), T.int64(1)), "float32"), lv2: T.Buffer((T.int64(2), T.int64(1)), "float32"), T_add: T.Buffer((T.int64(2), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(2), T.int64(1)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(matmul[v_ax0, v_ax1], lv2[v_ax0, v_ax1])
                T.writes(T_add[v_ax0, v_ax1])
                T_add[v_ax0, v_ax1] = matmul[v_ax0, v_ax1] + lv2[v_ax0, v_ax1]

    @T.prim_func(private=True)
    def matmul(lv: T.Buffer((T.int64(2), T.int64(2)), "float32"), permute_dims: T.Buffer((T.int64(2), T.int64(1)), "float32"), matmul: T.Buffer((T.int64(2), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        for i0, i1, k in T.grid(T.int64(2), T.int64(1), T.int64(2)):
            with T.block("matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(lv[v_i0, v_k], permute_dims[v_k, v_i1])
                T.writes(matmul[v_i0, v_i1])
                with T.init():
                    matmul[v_i0, v_i1] = T.float32(0.0)
                matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + lv[v_i0, v_k] * permute_dims[v_k, v_i1]

    @T.prim_func(private=True)
    def reshape(fc1_weight: T.Buffer((T.int64(2), T.int64(2)), "float32"), T_reshape: T.Buffer((T.int64(2), T.int64(1), T.int64(2)), "float32")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(1), T.int64(2)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(fc1_weight[(v_ax2 // T.int64(2) + v_ax0 + v_ax1) % T.int64(2), v_ax2 % T.int64(2)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = fc1_weight[(v_ax2 // T.int64(2) + v_ax0 + v_ax1) % T.int64(2), v_ax2 % T.int64(2)]

    @T.prim_func(private=True)
    def reshape1(fc1_bias: T.Buffer((T.int64(2), T.int64(2)), "float32"), T_reshape: T.Buffer((T.int64(2), T.int64(2), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(2), T.int64(1)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(fc1_bias[((v_ax1 + v_ax2) // T.int64(2) + v_ax0) % T.int64(2), (v_ax1 + v_ax2) % T.int64(2)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = fc1_bias[((v_ax1 + v_ax2) // T.int64(2) + v_ax0) % T.int64(2), (v_ax1 + v_ax2) % T.int64(2)]

    @T.prim_func(private=True)
    def transpose(lv1: T.Buffer((T.int64(2), T.int64(2), T.int64(1)), "float32"), T_transpose: T.Buffer((T.int64(2), T.int64(2), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(2), T.int64(1)):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv1[v_ax1, v_ax0, v_ax2])
                T.writes(T_transpose[v_ax0, v_ax1, v_ax2])
                T_transpose[v_ax0, v_ax1, v_ax2] = lv1[v_ax1, v_ax0, v_ax2]

    @T.prim_func(private=True)
    def transpose1(lv1: T.Buffer((T.int64(1), T.int64(2)), "float32"), T_transpose: T.Buffer((T.int64(2), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(2), T.int64(1)):
            with T.block("T_transpose"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv1[v_ax1, v_ax0])
                T.writes(T_transpose[v_ax0, v_ax1])
                T_transpose[v_ax0, v_ax1] = lv1[v_ax1, v_ax0]

    @R.function
    def forward(x: R.Tensor((2, 2), dtype="float32"), fc1_weight: R.Tensor((2, 2), dtype="float32"), fc1_bias: R.Tensor((2, 2), dtype="float32")) -> R.Tensor((2, 1), dtype="float32"):
        R.func_attr({"num_input": 1})
        cls = Module
        with R.dataflow():
            lv = R.call_dps_packed("runtime.disco.broadcast_from_worker0", (x, R.prim_value(0)), out_sinfo=R.Tensor((2, 2), dtype="float32"))
            lv_1 = R.call_tir(cls.reshape, (fc1_weight,), out_sinfo=R.Tensor((2, 1, 2), dtype="float32"))
            lv1 = R.call_dps_packed("runtime.disco.scatter_from_worker0", (lv_1, R.prim_value(0)), out_sinfo=R.Tensor((1, 2), dtype="float32"))
            lv1_1 = R.call_tir(cls.reshape1, (fc1_bias,), out_sinfo=R.Tensor((2, 2, 1), dtype="float32"))
            lv2 = R.call_tir(cls.transpose, (lv1_1,), out_sinfo=R.Tensor((2, 2, 1), dtype="float32"))
            lv2_1 = R.call_dps_packed("runtime.disco.scatter_from_worker0", (lv2, R.prim_value(0)), out_sinfo=R.Tensor((2, 1), dtype="float32"))
            permute_dims = R.call_tir(cls.transpose1, (lv1,), out_sinfo=R.Tensor((2, 1), dtype="float32"))
            matmul = R.call_tir(cls.matmul, (lv, permute_dims), out_sinfo=R.Tensor((2, 1), dtype="float32"))
            _add = R.call_tir(cls.add, (matmul, lv2_1), out_sinfo=R.Tensor((2, 1), dtype="float32"))
            # 我手動增加這行
            _gather = R.call_dps_packed("runtime.disco.allgather", [_add, True], out_sinfo=R.Tensor((2, 2), dtype="float32"))
            # 原本（2,1)手動改成（2,2)
            gv: R.Tensor((2, 2), dtype="float32") = _gather
            R.output(gv)
        return gv