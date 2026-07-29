import argparse
import numpy as np
import tvm
from tvm.runtime import disco
import os
from tvm import relax as rx
from tvm.script import relax as R



N = 4  
@tvm.script.ir_module
class CCLOps:
    @R.function
    def matmul_shard(
        x: R.Tensor((4, 2), "float32"),
        w_shard: R.Tensor((2, 1), "float32"),
    ) -> R.Tensor((4, 1), "float32"):
        R.func_attr({"global_symbol": "matmul_shard"})
        with R.dataflow():
            # (4,2) @ (2,1) = (4,1) -> 4 elements, divisible by 4 workers (allreduce-safe)
            q: R.Tensor((4, 1), "float32") = R.matmul(x, w_shard)
            R.output(q)
        return q




parser = argparse.ArgumentParser()
parser.add_argument("--host", default="192.168.50.169")
parser.add_argument("--port", type=int, default=18000)
args = parser.parse_args()

print("==== Setup Session ====")
sess = disco.SocketSession(N, 1, 1, args.host, args.port, True)
sess.init_ccl("cpuccl", *range(N))
sess._sync_all()

dev = tvm.cpu(0)
target = tvm.target.Target("llvm")
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ccl_test.so")

tvm.compile(rx.get_pipeline("zero")(CCLOps), target=target).export_library(path)
sess.upload_vm_module(path)
sess._sync_all()


X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype="float32")  # 4×2
W = np.array([[1, 0, 1, 0], 
              [0, 1, 0, 1]], dtype="float32")       # 2×4


# ---- broadcast X : full X -> every worker ----
d_X = sess.broadcast(X)
sess._sync_all()
print("==== after broadcast(X) ====")
for w in range(N):
    got = d_X.debug_get_from_remote(w).numpy()
    print(f"  worker{w} received X =\n{got}")
sess._sync_all()
# ---- scatter W columns : worker i gets column i as (2,1) ----
W_shards = W.T.reshape(N, 2, 1).copy()
d_Wsh = sess.scatter(W_shards)
sess._sync_all()
print("==== after scatter(W) ====")
for w in range(N):
    got = d_Wsh.debug_get_from_remote(w).numpy().ravel().tolist()
    print(f"  worker{w} received W-col = {got}  (expect {W[:, w].tolist()})")


mod = sess.load_vm_module(path); 
d_q = mod["matmul_shard"](d_X, d_Wsh)                      # (2,1) per worker
for w in range(N):
    got = d_q.debug_get_from_remote(w).numpy().ravel().tolist()
    print(f"worker{w}: X@W[:,{w}] = {got}  (expect {(X @ W[:, w:w+1]).ravel().tolist()})")

# ---- allreduce sum/prod/max/min/avg on the per-worker (4,1) results ----
for name in ("sum", "prod", "max", "min","avg"):
    d_out = sess.empty((4, 1), "float32")
    sess.allreduce(d_q, d_out, name)
    sess._sync_all()
    print(f"allreduce {name:4s} =\n{d_out.debug_get_from_remote(0).numpy().ravel()}")

# ---- allgather : each (4,1) -> everyone gets (16,1) = all four workers' results ----
d_ag = sess.empty((N * 4, 1), "float32")
sess.allgather(d_q, d_ag)
sess._sync_all()
for w in range(N):
    print(f"allgather worker{w}:\n{d_ag.debug_get_from_remote(w).numpy().reshape(N,4)}") 

# ---- gather_to_worker0 : shards -> only w0 gets (16,1) ----
d_gw0 = sess.empty((N * 4, 1), "float32", worker0_only=True)
sess.gather_to_worker0(d_q, d_gw0)
sess._sync_all()
print("gather->w0:\n", d_gw0.debug_get_from_remote(0).numpy().reshape(N, 4))

sess._sync_all(); sess.shutdown()