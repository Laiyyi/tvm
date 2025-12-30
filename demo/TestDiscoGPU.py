#  這邊是拿simple model 示範跑 disco 單張gpu 的範例 ***
import tvm
from tvm import relax
from tvm.relax.frontend import nn
from tvm import dlight as dl

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
#mod = relax.get_pipeline("zero")(mod)
# 以上優化

# build and deployment
import numpy as np
from tvm.runtime import disco



devices = [0]
#disco.ThreadedSession, diso.ProcessSession
sess = disco.ProcessSession(num_workers=len(devices))
sess.init_ccl("nccl", *devices)

import os
path = os.path.join(os.path.dirname(__file__), "test.so")

# 可以寫 dev = tvm.device("cpu",0)
dev = tvm.cuda(0)

# model inference 可以在這裡編譯，跟TestSimpleModel不一樣的是主要編譯改在這邊呼叫然後沒有export_library
# 要測試 gpu 改用 dev = tvm.cuda(0) 但是前面就沒辦法用 mod = relax.get_pipeline("zero")(mod)
# 必須要用這個 relax_build function
# 還要多加這行
# 多加這行
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
    
# TestSimpleModel 是直接 vm = relax.VirtualMachine(ex, dev) 再來 runtime.tensor 設定好資料執行
mod = sess.load_vm_module(path)

# 隨機生成
data_np = np.random.randn(1, 784).astype("float32")
# nnLinear 內的實做會把兩個做 transpose 所以 weight shape 要反過來
fc1weight_np = np.random.randn(256, 784).astype("float32")
fc1bias_np   = np.random.randn(256,).astype("float32")

fc2weight_np = np.random.randn(256, 10).astype("float32")
fc2bias_np   = np.random.randn(10,).astype("float32")

# copy 到 worker 0
data = sess.broadcast(data_np)
fc1weight = sess.copy_to_worker_0(fc1weight_np)
fc1bias = sess.copy_to_worker_0(fc1bias_np)
fc2weight = sess.copy_to_worker_0(fc2weight_np)
fc2bias = sess.copy_to_worker_0(fc2bias_np)


dref_output = mod["forward"](data, fc1weight, fc1bias, fc2weight, fc2bias)

output_strorage = tvm.runtime.empty((1, 10), "float32", device=dev)
sess.copy_from_worker_0(output_strorage, dref_output)
sess.sync_worker_0()

print("output_strorage:", output_strorage.numpy())

