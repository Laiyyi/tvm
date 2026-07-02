# 這份是用來教學
import tvm
from tvm import relax
from tvm.relax.frontend import nn
from tvm.runtime import disco
from tvm import dlight as dl
import numpy as np
import os


class SimpleModule(nn.Module):
    def __init__(self):
        super(SimpleModule, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x
    
mod, param_spec = SimpleModule().export_tvm(
    spec={"forward": {"x": nn.spec.Tensor((1, 784), "float32")}}
)


# @I.ir_module
# class SimpleModule:
#     @R.function
#     def forward(x: R.Tensor((1, 784), dtype="float32"), 
#                 fc1_weight: R.Tensor((256, 784), dtype="float32"), 
#                 fc1_bias: R.Tensor((256,), dtype="float32"), 
#                 fc2_weight: R.Tensor((10, 256), dtype="float32"), 
#                 fc2_bias: R.Tensor((10,), dtype="float32")) -> R.Tensor((1, 10), dtype="float32"):
#         R.func_attr({"num_input": 1})
#         with R.dataflow():
#             permute_dims: R.Tensor((784, 256), dtype="float32") = R.permute_dims(fc1_weight, axes=None)
#             matmul: R.Tensor((1, 256), dtype="float32") = R.matmul(x, permute_dims, out_dtype="void")
#             add: R.Tensor((1, 256), dtype="float32") = R.add(matmul, fc1_bias)
#             relu: R.Tensor((1, 256), dtype="float32") = R.nn.relu(add)
#             permute_dims1: R.Tensor((256, 10), dtype="float32") = R.permute_dims(fc2_weight, axes=None)
#             matmul1: R.Tensor((1, 10), dtype="float32") = R.matmul(relu, permute_dims1, out_dtype="void")
#             add1: R.Tensor((1, 10), dtype="float32") = R.add(matmul1, fc2_bias)
#             gv: R.Tensor((1, 10), dtype="float32") = add1
#             R.output(gv)
#         return gv


path = os.path.join(os.path.dirname(__file__), "test.so")

dev = tvm.cuda(0)
target = tvm.target.Target.from_device(dev)
def relax_build(mod, target):
        with target:
            mod = relax.get_pipeline("zero")(mod)  # pylint: disable=no-value-for-parameter
            mod = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
                dl.gpu.Matmul(),
                dl.gpu.GEMV(),
                dl.gpu.Reduction(),
                dl.gpu.GeneralReduction(),
                dl.gpu.Fallback(),
            )(mod)
            return tvm.compile(mod, target=target)
ex = relax_build(mod, target).export_library(path)

devices = [0, 1]
#disco.ThreadedSession, diso.ProcessSession, disco.SocketSession
sess = disco.SocketSession(num_nodes=len(devices), 
                           num_workers_per_node=1,
                           num_groups=1, host="localhost", port=9091)

sess.init_ccl("nccl", *devices)
mod = sess.load_vm_module(path)



# 隨機生成
X = np.random.randn(128, 128).astype("float32")
W1 = np.random.randn(128, 128).astype("float32")
W2 = np.random.randn(128, 128).astype("float32")


data = sess.broadcast(X)
dW1 = sess.scatter(W1)
dW2 = sess.scatter(W2)



dref_output = mod["forward"](data, dW1, dW2)
output_strorage = tvm.runtime.empty((128, 128), "float32", device=dev)
sess.copy_from_worker_0(output_strorage, dref_output)
sess.sync_worker_0()

print("output_strorage:", output_strorage.numpy())


mod = relax.get_pipeline("zero")(mod)
ex = tvm.compile(mod, target="llvm")
dev = tvm.device("cuda",0)

vm = relax.VirtualMachine(ex, dev)

#準備資料
data = np.random.rand(1, 784).astype("float32")

#runtime規定用tensor他們內部統一個格式
#所以這邊就是把data轉成他們要的
tvm_data = tvm.runtime.tensor(data, device=dev)

params = [np.random.rand(*param.shape).astype("float32") for _, param in param_spec]
params = [tvm.runtime.tensor(param, device=dev) for param in params]
# param_spec 裡面其實是：
# fc1.weight [256, 784]
# fc1.bias [256]
# fc2.weight [10, 256]
# fc2.bias [10]
# for name, param in param_spec:
#     print("name =", name)
#     print("shape =", param.shape)
#     print("dtype =", param.dtype)
#     print("-------")
# _=fc1.weight fc1.weight fc1.weight 但是 np.randome 不需要才用_
# param才是參數資訊 shape dtype 等等

# 執行 forward
print(vm["forward"](tvm_data, *params).numpy())