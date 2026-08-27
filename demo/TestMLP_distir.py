"""Smallest model that exercises the full DistIR pipeline: 2-layer MLP, TP=2."""
import tvm
from tvm import relax
from tvm.script.parser import ir as I  # NOT tvm.script — device_mesh only lives here
from tvm.script.parser import relax as R


@I.ir_module(s_tir=True)
class MLP:
    I.module_attrs({"device_num": 2})
    I.module_global_infos({"mesh": [R.device_mesh((2,), I.Range(0, 2))]})

    @R.function
    def main(
        x: R.Tensor((128, 128), "float32"),
        weight1: R.Tensor((128, 128), "float32"),
        weight2: R.Tensor((128, 128), "float32"),
    ) -> R.Tensor((128, 128), "float32"):
        lv0 = R.matmul(x, weight1)
        lv1 = R.nn.gelu(lv0)
        lv2 = R.dist.annotate_sharding(lv1, device_mesh="mesh[0]", placement="S[1]")
        lv3 = R.matmul(lv2, weight2)
        return lv3


T = relax.distributed.transform
mod = MLP
for name, p in [
    ("PropagateSharding", T.PropagateSharding()),
    ("LowerGlobalViewToLocalView", T.LowerGlobalViewToLocalView()),
    ("LegalizeRedistribute", T.LegalizeRedistribute()),
    ("LowerDistIR", T.LowerDistIR()),
]:
    mod = p(mod)
    print(f"===== after {name} =====")
    mod.show()
print("OK")
