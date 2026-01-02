#這一份不要刪掉，這是示範 import model 轉成IR 再用 relax 優化 compiler後 給 data 使用 vm與runtime執行

import tvm
from tvm import relax
from tvm.relax.frontend import nn


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

# 以下是轉換成 IR Module (mod.show()印出)
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


mod = relax.get_pipeline("zero")(mod)

# build and deployment 
import numpy as np

# Target部份也可以寫 target = tvm.target.Target("nvidia/geforce-rtx-3090-ti")
# cpu 要寫 llvm Gpu 要寫 cuda
# compile 裡面就有 build
ex = tvm.compile(mod, target="llvm")

# 可以寫 tvm = tvm.cpu()
dev = tvm.device("cpu",0)

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
