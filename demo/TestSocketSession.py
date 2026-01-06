import numpy as np
import os
from tvm.runtime import disco

def get_free_port():
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    print(f"Found free port: {port}")
    s.close()
    return port




devices = [0,1]
num_workers = len(devices)
num_nodes = len(devices)
num_groups = 1
assert num_workers % num_nodes == 0
num_workers_per_node = num_workers // num_nodes
server_host = "localhost"
server_port = get_free_port()




sess = disco.SocketSession(
                num_nodes, num_workers_per_node, num_groups, server_host, server_port
            )

sess.shutdown()
# sess.init_ccl("mpi", *devices)
# after = sess.load_vm_module(path)
# dev = tvm.device("cpu",0)
