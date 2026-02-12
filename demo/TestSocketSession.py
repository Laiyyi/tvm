import numpy as np
import os
from tvm.runtime import disco
# mpiexec --host node1:4,node2:4,node3:4,node4:4 -n 16 /home/ubuntu/tvm-env/bin/python3 TestSocketSession.py
# 當執行這段後，再加上 from mpi4py import MPI 就已經啟動 MPI_Init
# from mpi4py import MPI
# 可以直接在後端c++初始化 前端不用



devices = [0,1]


sess = disco.MPISession()
# sess.init_ccl("mpi", *devices)


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

