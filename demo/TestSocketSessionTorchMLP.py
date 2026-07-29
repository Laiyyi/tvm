import argparse
import os
import numpy as np

import tvm
from tvm import relax as rx
from tvm.runtime import disco
from tvm.script import relax as R

@tvm.script.ir_module
class CCLOps:
    @R.function
    def test_bcast_scatter_matmul(
        x: R.Tensor((4, 2), "float32"),
        W: R.Tensor((2, 4), "float32"),
    ) -> R.Tensor((4, 1), "float32"):
        R.func_attr({"global_symbol": "test_bcast_scatter_matmul"})
        with R.dataflow():
            broadcast_x: R.Tensor((4, 2), "float32") = R.ccl.broadcast_from_worker0(x)
            scattered_W: R.Tensor((2, 1), "float32") = R.ccl.scatter_from_worker0(W, 4, axis=1)
            # (4,2) @ (2,1) = (4,1) -> 4 elements, divisible by 4 workers (allreduce-safe)
            result: R.Tensor((4, 1), "float32") = R.matmul(broadcast_x, scattered_W)
            R.output(result)
        return result

    @R.function
    def test_allreduce(
        x: R.Tensor((4, 1), "float32"),
    ) -> R.Tensor((4, 1), "float32"):
        R.func_attr({"global_symbol": "test_allreduce"})
        with R.dataflow():
            result: R.Tensor((4, 1), "float32") = R.ccl.allreduce(x, "sum")
            R.output(result)
        return result

    @R.function
    def test_allgather(
        x: R.Tensor((4, 1), "float32"),
    ) -> R.Tensor((16, 1), "float32"):
        R.func_attr({"global_symbol": "test_allgather"})
        with R.dataflow():
            # allgather stacks along axis 0: (4,1)×4 -> (16,1)
            result: R.Tensor((16, 1), "float32") = R.ccl.allgather(x, 4)
            R.output(result)
        return result

    @R.function
    def test_gather_to_worker0(
        x: R.Tensor((4, 1), "float32"),
    ) -> R.Tensor((16, 1), "float32"):
        R.func_attr({"global_symbol": "test_gather_to_worker0"})
        with R.dataflow():
            # gather_to_worker0 stacks along axis 0: (4,1)×4 -> (16,1) on w0
            result: R.Tensor((16, 1), "float32") = R.ccl.gather_to_worker0(x, 4)
            R.output(result)
        return result


parser = argparse.ArgumentParser()
parser.add_argument("--num-nodes", type=int, default=4)
parser.add_argument("--num-workers-per-node", type=int, default=1)
parser.add_argument("--host", default="192.168.50.169")
# parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=18000)
parser.add_argument("--build-ring", type=lambda s: s.lower() in ("1", "true", "yes"), default=True)
args = parser.parse_args()

num_workers = args.num_nodes * args.num_workers_per_node
devices = list(range(num_workers))

print("==== Setup Session ====")
sess = disco.SocketSession(args.num_nodes, args.num_workers_per_node, 1,
                           args.host, args.port, args.build_ring)
sess.init_ccl("cpuccl", *devices)
sess._sync_all()

print("==== Prepare Data ====")
X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype="float32")  # 4×2
W = np.array([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]], dtype="float32")        # 2×4

print(f"X = \n{X}")
print(f"W = \n{W}")

# Upload & compile
dev = tvm.cpu(0)
target = tvm.target.Target("llvm")
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ccl_test.so")

CCLMod = rx.get_pipeline("zero")(CCLOps)
tvm.compile(CCLMod, target=target).export_library(path)

sess.upload_vm_module(path)
sess._sync_all()
CCLMod = sess.load_vm_module(path)
sess._sync_all()

print("\n==== Step 1: broadcast X + scatter W + matmul ====")
d_X = sess.empty((4, 2), "float32")
d_W = sess.empty((2, 4), "float32")
d_X.debug_copy_from(0, X)  # only w0 has X
d_W.debug_copy_from(0, W)  # only w0 has full W

d_matmul = CCLMod["test_bcast_scatter_matmul"](d_X, d_W)
matmul_result = tvm.runtime.empty((4, 1), "float32", device=dev)
sess.copy_from_worker_0(matmul_result, d_matmul)
sess._sync_all()
matmul_result = matmul_result.numpy()

print(f"Each worker computes: X @ W_shard (4,2) @ (2,1) = (4,1)")
print(f"Worker 0 result:\n{matmul_result}")

# Worker 0 gets column 0 of W: X @ W[:, 0:1]
expected_w0 = np.dot(X, W[:, 0:1])
print(f"Expected (w0 column):\n{expected_w0}")
print(f"Match: {np.allclose(matmul_result, expected_w0, atol=1e-4)}\n")

print("==== Step 2: allreduce (sum of 4 workers) ====")
# 4 workers each send their matmul result, sum them
d_local = sess.empty((4, 1), "float32")
for i in range(4):
    # Each worker i has X @ W[:,i:i+1]  -> (4,1)
    local_val = np.dot(X, W[:, i:i+1])
    d_local.debug_copy_from(i, local_val)
sess._sync_all()

d_sum = CCLMod["test_allreduce"](d_local)
sum_result = tvm.runtime.empty((4, 1), "float32", device=dev)
sess.copy_from_worker_0(sum_result, d_sum)
sess._sync_all()
sum_result = sum_result.numpy()

expected_sum = np.dot(X, W).sum(axis=1, keepdims=True)  # (4,1)
print(f"Sum result (all 4 workers):\n{sum_result}")
print(f"Expected (X @ W sum over columns):\n{expected_sum}")
print(f"Match: {np.allclose(sum_result, expected_sum, atol=1e-4)}\n")

print("==== Step 3: allgather ====")
d_shard = sess.empty((4, 1), "float32")
for i in range(4):
    shard_val = np.dot(X, W[:, i:i+1])
    d_shard.debug_copy_from(i, shard_val)
sess._sync_all()

d_gather = CCLMod["test_allgather"](d_shard)
gather_result = tvm.runtime.empty((16, 1), "float32", device=dev)
sess.copy_from_worker_0(gather_result, d_gather)
sess.sync_worker_0()
gather_result = gather_result.numpy().reshape(4, 4)  # row i = worker i's (X@W[:,i])

# allgather stacks each worker's (4,1) column-result as rows -> (X@W).T
expected_gather = np.dot(X, W).T  # (4,4)
print(f"Allgather result (reshaped 4×4):\n{gather_result}")
print(f"Expected (X @ W).T:\n{expected_gather}")
print(f"Match: {np.allclose(gather_result, expected_gather, atol=1e-4)}\n")

print("==== Step 4: gather_to_worker0 ====")
d_gather_local = sess.empty((4, 1), "float32")
for i in range(4):
    local_val = np.dot(X, W[:, i:i+1])
    d_gather_local.debug_copy_from(i, local_val)
sess._sync_all()

d_final = CCLMod["test_gather_to_worker0"](d_gather_local)
final_result = tvm.runtime.empty((16, 1), "float32", device=dev)
sess.copy_from_worker_0(final_result, d_final)
sess.sync_worker_0()
final_result = final_result.numpy().reshape(4, 4)  # row i = worker i's (X@W[:,i])

expected_final = np.dot(X, W).T  # (4,4)
print(f"Gather_to_worker0 result (reshaped 4×4):\n{final_result}")
print(f"Expected (X @ W).T:\n{expected_final}")
print(f"Match: {np.allclose(final_result, expected_final, atol=1e-4)}\n")

print("==== Shutdown ====")
sess._sync_all()
sess.shutdown()
print("Done!")