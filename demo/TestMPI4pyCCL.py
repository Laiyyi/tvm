# 用來測試光靠 python 的 mpi4py 是否就能達到 cpu 的 cluster
# 答案是可以....
import numpy as np
import os
# mpiexec --host node1,node2,node3,node4 -n 4 /home/buntu/tvm-env/bin/python3 TestMPI4pyCCL.py
from mpi4py import MPI
import tvm
from tvm import relax

from tvm.script.parser import ir as I
from tvm.script.parser import relax as R
@I.ir_module
class Module:
    I.module_attrs({"device_num" : 2})
    I.module_global_infos(
        {
            "mesh": [
                R.device_mesh((2,), I.Range(0, 2))
            ]
        }
    )

    @R.function
    def forward(
        x: R.Tensor((1, 784), dtype="float32"), 
        fc1_weight: R.Tensor((128, 784), dtype="float32"), 
        fc1_bias: R.Tensor((128,), dtype="float32"),) -> R.Tensor((1, 128), dtype="float32"):
        
        R.func_attr({"num_input": 1})
        with R.dataflow():
            permute_dims: R.Tensor((784, 128), dtype="float32") = R.permute_dims(fc1_weight, axes=None)
            matmul: R.Tensor((1, 128), dtype="float32") = R.matmul(x, permute_dims, out_dtype="void")
            add: R.Tensor((1, 128), dtype="float32") = R.add(matmul, fc1_bias)
            gv: R.Tensor((1, 128), dtype="float32") = add
            R.output(gv)
        return gv

afterlowerdistir = tvm.relax.transform.LegalizeOps()(Module)
afterlowerdistir.show()
after = relax.get_pipeline("zero")(afterlowerdistir)
ex = tvm.compile(after, target="llvm")


dev = tvm.device("cpu",0)

vm = relax.VirtualMachine(ex, dev)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 只有 rank 0 生成原始資料
if rank == 0:
   
    data_np = np.ones((1, 784), dtype="float32")
    fc1weight_np = np.ones((256, 784), dtype="float32")
    fc1bias_np   = np.ones((256,), dtype="float32")
else:
    data_np = None
    fc1weight_np = None
    fc1bias_np = None

# -----------------------
# 1. Broadcast data_np
# -----------------------
from tvm.runtime._tensor import tensor as _as_Tensor

data_np = comm.bcast(data_np, root=0)
data_np_to_tensor = _as_Tensor(data_np)


# -----------------------
# 2. Scatter fc1weight and fc1bias
# -----------------------
# 計算每個 rank 拿多少行
rows_per_rank = 256 // size
remainder = 256 % size

# rank 0 分割資料
if rank == 0:
    weight_chunks = []
    bias_chunks = []
    start = 0
    for i in range(size):
        end = start + rows_per_rank + (1 if i < remainder else 0)
        weight_chunks.append(fc1weight_np[start:end])
        bias_chunks.append(fc1bias_np[start:end])
        start = end
else:
    weight_chunks = None
    bias_chunks = None

# scatter
fc1weight_local = comm.scatter(weight_chunks, root=0)
fc1weight_tensor = _as_Tensor(fc1weight_local)
fc1bias_local   = comm.scatter(bias_chunks, root=0)
fc1bias_local_tensor = _as_Tensor(fc1bias_local)

out_np = vm["forward"](data_np_to_tensor, fc1weight_tensor, fc1bias_local_tensor).numpy()
np.savetxt("fc1_output.txt", out_np, fmt="%.6f")
