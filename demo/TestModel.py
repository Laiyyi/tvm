#  這邊是試試看 model 不要用 relax 優化 看可不可以通過編譯build起來
import tvm
from tvm import relax
from tvm.relax.frontend import nn

import os

# m = nn.Linear(20, 30)
# input = torch.randn(128, 20)

# class SimpleModule(nn.Module):
#     def __init__(self):
#         super(SimpleModule, self).__init__()
#         self.fc1 = nn.Linear(784, 256)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(256, 10)
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu1(x)
#         x = self.fc2(x)
#         return x
    
# mod, param_spec = SimpleModule().export_tvm(
#     spec={"forward": {"x": nn.spec.Tensor((1, 784), "float32")}}
# )
# mod.show()
from tvm.script.parser import ir as I
from tvm.script.parser import relax as R
@I.ir_module
class Module:
    I.module_attrs({"device_num" : 4})
    I.module_global_infos(
        {
            "mesh": [
                R.device_mesh((4,), I.Range(0, 4))
            ]
        }
    )

    @R.function
    def forward(
        x: R.Tensor((1, 784), dtype="float32"), 
        fc1_weight: R.Tensor((256, 784), dtype="float32"), 
        fc1_bias: R.Tensor((256,), dtype="float32"), 
        fc2_weight: R.Tensor((10, 256), dtype="float32"), 
        fc2_bias: R.Tensor((10,), dtype="float32")) -> R.Tensor((1, 10), dtype="float32"):
        
        R.func_attr({"num_input": 1})
        with R.dataflow():
            permute_dims: R.Tensor((784, 256), dtype="float32") = R.permute_dims(fc1_weight, axes=None)
            permute_dims: R.Tensor((784, 256), dtype="float32") = R.distributed.annotate_sharding(permute_dims, 
                                                                                                  device_mesh="mesh[0]", placement="S[1]")
            matmul: R.Tensor((1, 256), dtype="float32") = R.matmul(x, permute_dims, out_dtype="void")
            add: R.Tensor((1, 256), dtype="float32") = R.add(matmul, fc1_bias)
            relu: R.Tensor((1, 256), dtype="float32") = R.nn.relu(add)
            permute_dims1: R.Tensor((256, 10), dtype="float32") = R.permute_dims(fc2_weight, axes=None)
            matmul1: R.Tensor((1, 10), dtype="float32") = R.matmul(relu, permute_dims1, out_dtype="void")
            add1: R.Tensor((1, 10), dtype="float32") = R.add(matmul1, fc2_bias)
            gv: R.Tensor((1, 10), dtype="float32") = add1
            R.output(gv)
        return gv
    
after = relax.distributed.transform.PropagateSharding()(Module)
after.show()

afterlowerdistir = relax.distributed.transform.LowerDistIR()(after)
afterlowerdistir.show()

afterlowerdistir = tvm.relax.transform.LegalizeOps()(afterlowerdistir)
afterlowerdistir.show()

after = relax.get_pipeline("zero")(afterlowerdistir)
path = os.path.join(os.path.dirname(__file__), "testmodel.so")
ex = tvm.compile(after, target="llvm").export_library(path)

