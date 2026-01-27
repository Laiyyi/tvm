import numpy as np
import os
from tvm.runtime import disco
# mpiexec --host node1:4,node2:4,node3:4,node4:4 -n 16 /home/buntu/tvm-env/bin/python3 TestSocketSession.py
# 當執行這段後，再加上 from mpi4py import MPI 就已經啟動 MPI_Init
from mpi4py import MPI


devices = [0, 1, 2, 3]

comm = MPI.COMM_WORLD
rank = comm.Get_rank()



if rank == 0:
    data_np      = np.random.randn(1, 784).astype("float32")
    fc1weight_np = np.random.randn(256, 784).astype("float32")
    fc1bias_np   = np.random.randn(256,).astype("float32")
    fc2weight_np = np.random.randn(256, 10).astype("float32")
    fc2bias_np   = np.random.randn(10,).astype("float32")
else:
    data_np      = np.empty((1, 784), dtype="float32")
    fc1weight_np = np.empty((256, 784), dtype="float32")
    fc1bias_np   = np.empty((256,), dtype="float32")
    fc2weight_np = np.empty((256, 10), dtype="float32")
    fc2bias_np   = np.empty((10,), dtype="float32")

comm.Bcast(data_np, root=0)
comm.Bcast(fc1weight_np, root=0)
comm.Bcast(fc1bias_np, root=0)
comm.Bcast(fc2weight_np, root=0)
comm.Bcast(fc2bias_np, root=0)

out_file = f"rank_{rank}_params.txt"

with open(out_file, "w") as f:
    f.write("data_np:\n")
    f.write(np.array2string(data_np, threshold=20))
    f.write("\n\n")

    f.write("fc1weight_np:\n")
    f.write(np.array2string(fc1weight_np, threshold=20))
    f.write("\n\n")

    f.write("fc1bias_np:\n")
    f.write(np.array2string(fc1bias_np))
    f.write("\n\n")

    f.write("fc2weight_np:\n")
    f.write(np.array2string(fc2weight_np, threshold=20))
    f.write("\n\n")

    f.write("fc2bias_np:\n")
    f.write(np.array2string(fc2bias_np))
    f.write("\n")

# mod = sess.load_vm_module(path)
# copy 到 worker 0
# data = sess.broadcast(data_np)
# fc1weight = sess.copy_to_worker_0(fc1weight_np)
# fc1bias = sess.copy_to_worker_0(fc1bias_np)
# fc2weight = sess.copy_to_worker_0(fc2weight_np)
# fc2bias = sess.copy_to_worker_0(fc2bias_np)
# dref_output = mod["forward"](data, fc1weight, fc1bias, fc2weight, fc2bias)

# output_strorage = tvm.runtime.empty((1, 10), "float32", device=dev)
# sess.copy_from_worker_0(output_strorage, dref_output)
# sess.sync_worker_0()

# print("output_strorage:", output_strorage.numpy())

