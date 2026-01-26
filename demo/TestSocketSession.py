import numpy as np
import os
from tvm.runtime import disco

# mpiexec --host node1:4,node2:4,node3:4,node4:4 -n 16 /home/buntu/tvm-env/bin/python3 TestSocketSession.py
# 當執行這段後，再加上 from mpi4py import MPI 就已經啟動 MPI_Init
from mpi4py import MPI


devices = [0, 1, 2, 3]

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

sess = disco.ProcessSession(num_workers=1)

if rank == 0:
    sess.init_ccl("mpi", *devices)
    print(f"Rank 0")

else:
    print(f"OtherRank")


# after = sess.load_vm_module(path)
# dev = tvm.device("cpu",0)
