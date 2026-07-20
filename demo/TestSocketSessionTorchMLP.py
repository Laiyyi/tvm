import argparse
import os
import sys

import numpy as np

import tvm
from tvm import relax as rx
from tvm.runtime import disco
from tvm.script import relax as R
from tvm.runtime.vm import VirtualMachine

@tvm.script.ir_module
class Attention:  # pylint: disable=too-few-public-methods
        @R.function
        def main(  # pylint: disable=too-many-locals
            x: R.Tensor((1, 10, 128), "float32"),
            Wq: R.Tensor((128, 512), "float32"),
            Wk: R.Tensor((128, 512), "float32"),
            Wv: R.Tensor((128, 512), "float32"),
            Wo: R.Tensor((512, 128), "float32"),
        ) -> R.Tensor((128, 128), "float32"):
            R.func_attr({"global_symbol": "main"})
            with R.dataflow():
                # q
                lv0: R.Tensor((1, 10, 512), "float32") = R.matmul(x, Wq)
                lv1: R.Tensor((1, 10, 8, 64), "float32") = R.reshape(lv0, [1, 10, 8, 64])
                lv2: R.Tensor((1, 8, 10, 64), "float32") = R.permute_dims(lv1, [0, 2, 1, 3])
                # k
                lv3: R.Tensor((1, 10, 512), "float32") = R.matmul(x, Wk)
                lv4: R.Tensor((1, 10, 8, 64), "float32") = R.reshape(lv3, [1, 10, 8, 64])
                lv5: R.Tensor((1, 8, 10, 64), "float32") = R.permute_dims(lv4, [0, 2, 1, 3])
                # v
                lv6: R.Tensor((1, 10, 512), "float32") = R.matmul(x, Wv)
                lv7: R.Tensor((1, 10, 8, 64), "float32") = R.reshape(lv6, [1, 10, 8, 64])
                lv8: R.Tensor((1, 8, 10, 64), "float32") = R.permute_dims(lv7, [0, 2, 1, 3])
                # softmax(q @ k / sqrt(dk))
                lv9: R.Tensor((1, 8, 64, 10), "float32") = R.permute_dims(lv5, [0, 1, 3, 2])
                lv10: R.Tensor((1, 8, 10, 10), "float32") = R.matmul(lv2, lv9)
                lv11: R.Tensor((1, 8, 10, 10), "float32") = R.multiply(
                    lv10, R.const(1 / 8, "float32")
                )
                lv12: R.Tensor((1, 8, 10, 10), "float32") = R.nn.softmax(lv11, axis=-1)
                # attn_weight @ v
                lv13: R.Tensor((1, 8, 10, 64), "float32") = R.matmul(lv12, lv8)
                lv14: R.Tensor((1, 10, 8, 64), "float32") = R.permute_dims(lv13, [0, 2, 1, 3])
                lv15: R.Tensor((1, 10, 512), "float32") = R.reshape(lv14, [1, 10, 512])
                # attn_output @ o
                lv16: R.Tensor((1, 10, 128), "float32") = R.matmul(lv15, Wo)
                R.output(lv16)
            return lv16
        
@tvm.script.ir_module
class ShardedAttention:  # pylint: disable=too-few-public-methods
        @R.function
        def main(  # pylint: disable=too-many-locals
            x: R.Tensor((1, 10, 128), "float32"),
            Wq: R.Tensor((128, 256), "float32"),  # shard along axis 1
            Wk: R.Tensor((128, 256), "float32"),  # shard along axis 1
            Wv: R.Tensor((128, 256), "float32"),  # shard along axis 1
            Wo: R.Tensor((256, 128), "float32"),  # shard along axis 0
        ) -> R.Tensor((128, 128), "float32"):
            R.func_attr({"global_symbol": "main"})
            with R.dataflow():
                broadcast_x: R.Tensor((1, 10, 128), "float32") = R.ccl.broadcast_from_worker0(x)
                # q
                lv0: R.Tensor((1, 10, 256), "float32") = R.matmul(broadcast_x, Wq)
                lv1: R.Tensor((1, 10, 4, 64), "float32") = R.reshape(lv0, [1, 10, 4, 64])
                lv2: R.Tensor((1, 4, 10, 64), "float32") = R.permute_dims(lv1, [0, 2, 1, 3])
                # k
                lv3: R.Tensor((1, 10, 256), "float32") = R.matmul(broadcast_x, Wk)
                lv4: R.Tensor((1, 10, 4, 64), "float32") = R.reshape(lv3, [1, 10, 4, 64])
                lv5: R.Tensor((1, 4, 10, 64), "float32") = R.permute_dims(lv4, [0, 2, 1, 3])
                # v
                lv6: R.Tensor((1, 10, 256), "float32") = R.matmul(broadcast_x, Wv)
                lv7: R.Tensor((1, 10, 4, 64), "float32") = R.reshape(lv6, [1, 10, 4, 64])
                lv8: R.Tensor((1, 4, 10, 64), "float32") = R.permute_dims(lv7, [0, 2, 1, 3])
                # softmax(q @ k / sqrt(dk))
                lv9: R.Tensor((1, 4, 64, 10), "float32") = R.permute_dims(lv5, [0, 1, 3, 2])
                lv10: R.Tensor((1, 4, 10, 10), "float32") = R.matmul(lv2, lv9)
                lv11: R.Tensor((1, 4, 10, 10), "float32") = R.multiply(
                    lv10, R.const(1 / 8, "float32")
                )
                lv12: R.Tensor((1, 4, 10, 10), "float32") = R.nn.softmax(lv11, axis=-1)
                # attn_weight @ v
                lv13: R.Tensor((1, 4, 10, 64), "float32") = R.matmul(lv12, lv8)
                lv14: R.Tensor((1, 10, 4, 64), "float32") = R.permute_dims(lv13, [0, 2, 1, 3])
                lv15: R.Tensor((1, 10, 256), "float32") = R.reshape(lv14, [1, 10, 256])
                # attn_output @ o
                lv16: R.Tensor((1, 10, 128), "float32") = R.matmul(lv15, Wo)
                lv17: R.Tensor((1, 10, 128), "float32") = R.ccl.allreduce(lv16, "sum")
                R.output(lv17)
            return lv17
        


parser = argparse.ArgumentParser()
parser.add_argument("--num-nodes", type=int, default=2)
parser.add_argument("--num-workers-per-node", type=int, default=1)
parser.add_argument("--num-groups", type=int, default=1)
parser.add_argument("--host", default="192.168.50.169")
parser.add_argument("--port", type=int, default=18000)
parser.add_argument("--build-ring", type=lambda s: s.lower() in ("1", "true", "yes"), default=True)
args = parser.parse_args()

num_workers = args.num_nodes * args.num_workers_per_node
devices = list(range(num_workers))
print(f"---- Now Create SocketSession ---- ")
sess = disco.SocketSession(args.num_nodes, args.num_workers_per_node, args.num_groups,
                           args.host, args.port, args.build_ring)

print(f"---- Now Cpuccl start to initial ---- ")
sess.init_ccl("cpuccl", *devices)

print(f"---- Preparing dev/target/mod ---- ")
dev = tvm.cpu(0)
target = tvm.target.Target("llvm")
mod = rx.get_pipeline("zero")(Attention)

print(f"---- Preparing data---- ")
X = np.random.randn(1, 10, 128).astype("float32")
Wq = np.random.randn(128, 512).astype("float32")
Wk = np.random.randn(128, 512).astype("float32")
Wv = np.random.randn(128, 512).astype("float32")
Wo = np.random.randn(512, 128).astype("float32")
Y_expected = VirtualMachine(tvm.compile(mod, target=target), device=dev)["main"](
        tvm.runtime.tensor(X, device=dev),
        tvm.runtime.tensor(Wq, device=dev),
        tvm.runtime.tensor(Wk, device=dev),
        tvm.runtime.tensor(Wv, device=dev),
        tvm.runtime.tensor(Wo, device=dev),
    ).numpy()



d_X = sess.empty((1, 10, 128), "float32")
d_Wq = sess.empty((128, 256), "float32")
d_Wk = sess.empty((128, 256), "float32")
d_Wv = sess.empty((128, 256), "float32")
d_Wo = sess.empty((256, 128), "float32")

d_X.debug_copy_from(0, X)
d_Wq.debug_copy_from(0, Wq[:, :256])
d_Wq.debug_copy_from(1, Wq[:, 256:])
d_Wk.debug_copy_from(0, Wk[:, :256])
d_Wk.debug_copy_from(1, Wk[:, 256:])
d_Wv.debug_copy_from(0, Wv[:, :256])
d_Wv.debug_copy_from(1, Wv[:, 256:])
d_Wo.debug_copy_from(0, Wo[:256, :])
d_Wo.debug_copy_from(1, Wo[256:, :])



print(f"---- Preparing path and Shardmod---- ")
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.so")
Shardmod = rx.get_pipeline("zero")(ShardedAttention)

print(f"---- Exporting ---- ")
tvm.compile(Shardmod, target=target).export_library(path)
print(f"---- Path:{path} ---- ")


sess.upload_vm_module(path)
sess._sync_all() 

print(f"---- upload done---- ")
Shardmod = sess.load_vm_module(path)
print(f"---- load done---- ")
sess._sync_all() 



print(f"---- Main Start ---- ")
d_Y = Shardmod["main"](d_X, d_Wq, d_Wk, d_Wv, d_Wo)
Y_result = tvm.runtime.empty((1, 10, 128), "float32", device=dev)
sess.copy_from_worker_0(Y_result, d_Y)
print(f"---- Sync0 Start ---- ")
sess.sync_worker_0()
Y_result = Y_result.numpy()
    # pylint: enable=invalid-name
np.testing.assert_allclose(Y_result, Y_expected, rtol=1e-3, atol=1e-3)
print(f"---- shutdown ---- ")
sess._sync_all() 
sess.shutdown() 