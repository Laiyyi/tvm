import numpy as np
import os
from tvm.runtime import disco


from mpi4py import MPI
import subprocess
import sys


def get_comm_rank():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    return comm, rank



devices = [0,1]
num_workers = len(devices)
num_nodes = len(devices)
num_groups = 1
assert num_workers % num_nodes == 0
num_workers_per_node = num_workers // num_nodes
server_host = "localhost"
# server_host = "192.168.50.169"
server_port = 38597

comm, rank = get_comm_rank()

if rank == 0:
    sess = disco.SocketSession(num_nodes, num_workers_per_node, num_groups, server_host, server_port)
    sess.init_ccl("mpi", *devices)
    sess.shutdown()
else:
    cmd = "tvm.exec.disco_remote_socket_session"
    remote_nodes = []
    
    for _ in range(num_nodes - 1):
        remote_nodes.append(
        subprocess.Popen(
                    [
                        "python3",
                        "-m",
                        cmd,
                        server_host,
                        str(server_port),
                        str(num_workers_per_node),
                    ],
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                )
            )
    

# after = sess.load_vm_module(path)
# dev = tvm.device("cpu",0)
