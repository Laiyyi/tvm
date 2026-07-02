#不要刪,拿 resnet 測試
import torch
from torch.export import export
from torchvision.models.resnet import ResNet18_Weights, resnet18

torch_model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()

import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

weight_args  = (torch.randn(1, 3, 224, 224, dtype=torch.float32),)

with torch.no_grad():
    exported_program = export(
        torch_model,
        weight_args
    )
    mod = from_exported_program(exported_program, keep_params_as_input=True)
    
mod, params = relax.frontend.detach_params(mod)
#mod.show()

from tvm.script import ir as I
from tvm.script import relax as R

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
    def main(x: R.Tensor((1, 3, 224, 224), dtype="float32"), p_conv1_weight: R.Tensor((64, 3, 7, 7), dtype="float32"), p_bn1_weight: R.Tensor((64,), dtype="float32"), p_bn1_bias: R.Tensor((64,), dtype="float32"), p_layer1_0_conv1_weight: R.Tensor((64, 64, 3, 3), dtype="float32"), p_layer1_0_bn1_weight: R.Tensor((64,), dtype="float32"), p_layer1_0_bn1_bias: R.Tensor((64,), dtype="float32"), p_layer1_0_conv2_weight: R.Tensor((64, 64, 3, 3), dtype="float32"), p_layer1_0_bn2_weight: R.Tensor((64,), dtype="float32"), p_layer1_0_bn2_bias: R.Tensor((64,), dtype="float32"), p_layer1_1_conv1_weight: R.Tensor((64, 64, 3, 3), dtype="float32"), p_layer1_1_bn1_weight: R.Tensor((64,), dtype="float32"), p_layer1_1_bn1_bias: R.Tensor((64,), dtype="float32"), p_layer1_1_conv2_weight: R.Tensor((64, 64, 3, 3), dtype="float32"), p_layer1_1_bn2_weight: R.Tensor((64,), dtype="float32"), p_layer1_1_bn2_bias: R.Tensor((64,), dtype="float32"), p_layer2_0_conv1_weight: R.Tensor((128, 64, 3, 3), dtype="float32"), p_layer2_0_bn1_weight: R.Tensor((128,), dtype="float32"), p_layer2_0_bn1_bias: R.Tensor((128,), dtype="float32"), p_layer2_0_conv2_weight: R.Tensor((128, 128, 3, 3), dtype="float32"), p_layer2_0_bn2_weight: R.Tensor((128,), dtype="float32"), p_layer2_0_bn2_bias: R.Tensor((128,), dtype="float32"), p_layer2_0_downsample_0_weight: R.Tensor((128, 64, 1, 1), dtype="float32"), p_layer2_0_downsample_1_weight: R.Tensor((128,), dtype="float32"), p_layer2_0_downsample_1_bias: R.Tensor((128,), dtype="float32"), p_layer2_1_conv1_weight: R.Tensor((128, 128, 3, 3), dtype="float32"), p_layer2_1_bn1_weight: R.Tensor((128,), dtype="float32"), p_layer2_1_bn1_bias: R.Tensor((128,), dtype="float32"), p_layer2_1_conv2_weight: R.Tensor((128, 128, 3, 3), dtype="float32"), p_layer2_1_bn2_weight: R.Tensor((128,), dtype="float32"), p_layer2_1_bn2_bias: R.Tensor((128,), dtype="float32"), p_layer3_0_conv1_weight: R.Tensor((256, 128, 3, 3), dtype="float32"), p_layer3_0_bn1_weight: R.Tensor((256,), dtype="float32"), p_layer3_0_bn1_bias: R.Tensor((256,), dtype="float32"), p_layer3_0_conv2_weight: R.Tensor((256, 256, 3, 3), dtype="float32"), p_layer3_0_bn2_weight: R.Tensor((256,), dtype="float32"), p_layer3_0_bn2_bias: R.Tensor((256,), dtype="float32"), p_layer3_0_downsample_0_weight: R.Tensor((256, 128, 1, 1), dtype="float32"), p_layer3_0_downsample_1_weight: R.Tensor((256,), dtype="float32"), p_layer3_0_downsample_1_bias: R.Tensor((256,), dtype="float32"), p_layer3_1_conv1_weight: R.Tensor((256, 256, 3, 3), dtype="float32"), p_layer3_1_bn1_weight: R.Tensor((256,), dtype="float32"), p_layer3_1_bn1_bias: R.Tensor((256,), dtype="float32"), p_layer3_1_conv2_weight: R.Tensor((256, 256, 3, 3), dtype="float32"), p_layer3_1_bn2_weight: R.Tensor((256,), dtype="float32"), p_layer3_1_bn2_bias: R.Tensor((256,), dtype="float32"), p_layer4_0_conv1_weight: R.Tensor((512, 256, 3, 3), dtype="float32"), p_layer4_0_bn1_weight: R.Tensor((512,), dtype="float32"), p_layer4_0_bn1_bias: R.Tensor((512,), dtype="float32"), p_layer4_0_conv2_weight: R.Tensor((512, 512, 3, 3), dtype="float32"), p_layer4_0_bn2_weight: R.Tensor((512,), dtype="float32"), p_layer4_0_bn2_bias: R.Tensor((512,), dtype="float32"), p_layer4_0_downsample_0_weight: R.Tensor((512, 256, 1, 1), dtype="float32"), p_layer4_0_downsample_1_weight: R.Tensor((512,), dtype="float32"), p_layer4_0_downsample_1_bias: R.Tensor((512,), dtype="float32"), p_layer4_1_conv1_weight: R.Tensor((512, 512, 3, 3), dtype="float32"), p_layer4_1_bn1_weight: R.Tensor((512,), dtype="float32"), p_layer4_1_bn1_bias: R.Tensor((512,), dtype="float32"), p_layer4_1_conv2_weight: R.Tensor((512, 512, 3, 3), dtype="float32"), p_layer4_1_bn2_weight: R.Tensor((512,), dtype="float32"), p_layer4_1_bn2_bias: R.Tensor((512,), dtype="float32"), p_fc_weight: R.Tensor((1000, 512), dtype="float32"), p_fc_bias: R.Tensor((1000,), dtype="float32")) -> R.Tuple(R.Tensor((1, 1000), dtype="float32")):
        R.func_attr({"num_input": 1})
        with R.dataflow():
            lv: R.Tensor((1, 64, 112, 112), dtype="float32") = R.nn.conv2d(x, p_conv1_weight, strides=[2, 2], padding=[3, 3, 3, 3], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv1: R.Tuple(R.Tensor((1, 64, 112, 112), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = R.nn.batch_norm(lv, p_bn1_weight, p_bn1_bias, metadata["relax.expr.Constant"][0], metadata["relax.expr.Constant"][1], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv2: R.Tensor((1, 64, 112, 112), dtype="float32") = lv1[0]
            lv3: R.Tensor((1, 64, 112, 112), dtype="float32") = R.nn.relu(lv2)
            lv4: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.max_pool2d(lv3, pool_size=[3, 3], strides=[2, 2], dilation=[1, 1], padding=[1, 1, 1, 1], ceil_mode=False, count_include_pad=False, layout="NCHW", out_layout="NCHW")
            lv5: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.conv2d(lv4, p_layer1_0_conv1_weight, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv6: R.Tuple(R.Tensor((1, 64, 56, 56), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = R.nn.batch_norm(lv5, p_layer1_0_bn1_weight, p_layer1_0_bn1_bias, metadata["relax.expr.Constant"][2], metadata["relax.expr.Constant"][3], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv7: R.Tensor((1, 64, 56, 56), dtype="float32") = lv6[0]
            lv8: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.relu(lv7)
            lv9: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.conv2d(lv8, p_layer1_0_conv2_weight, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv10: R.Tuple(R.Tensor((1, 64, 56, 56), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = R.nn.batch_norm(lv9, p_layer1_0_bn2_weight, p_layer1_0_bn2_bias, metadata["relax.expr.Constant"][4], metadata["relax.expr.Constant"][5], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv11: R.Tensor((1, 64, 56, 56), dtype="float32") = lv10[0]
            lv12: R.Tensor((1, 64, 56, 56), dtype="float32") = R.add(lv11, lv4)
            lv13: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.relu(lv12)
            lv14: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.conv2d(lv13, p_layer1_1_conv1_weight, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv15: R.Tuple(R.Tensor((1, 64, 56, 56), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = R.nn.batch_norm(lv14, p_layer1_1_bn1_weight, p_layer1_1_bn1_bias, metadata["relax.expr.Constant"][6], metadata["relax.expr.Constant"][7], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv16: R.Tensor((1, 64, 56, 56), dtype="float32") = lv15[0]
            lv17: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.relu(lv16)
            lv18: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.conv2d(lv17, p_layer1_1_conv2_weight, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv19: R.Tuple(R.Tensor((1, 64, 56, 56), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = R.nn.batch_norm(lv18, p_layer1_1_bn2_weight, p_layer1_1_bn2_bias, metadata["relax.expr.Constant"][8], metadata["relax.expr.Constant"][9], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv20: R.Tensor((1, 64, 56, 56), dtype="float32") = lv19[0]
            lv21: R.Tensor((1, 64, 56, 56), dtype="float32") = R.add(lv20, lv13)
            lv22: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.relu(lv21)
            lv23: R.Tensor((1, 128, 28, 28), dtype="float32") = R.nn.conv2d(lv22, p_layer2_0_conv1_weight, strides=[2, 2], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv24: R.Tuple(R.Tensor((1, 128, 28, 28), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = R.nn.batch_norm(lv23, p_layer2_0_bn1_weight, p_layer2_0_bn1_bias, metadata["relax.expr.Constant"][10], metadata["relax.expr.Constant"][11], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv25: R.Tensor((1, 128, 28, 28), dtype="float32") = lv24[0]
            lv26: R.Tensor((1, 128, 28, 28), dtype="float32") = R.nn.relu(lv25)
            lv27: R.Tensor((1, 128, 28, 28), dtype="float32") = R.nn.conv2d(lv26, p_layer2_0_conv2_weight, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv28: R.Tuple(R.Tensor((1, 128, 28, 28), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = R.nn.batch_norm(lv27, p_layer2_0_bn2_weight, p_layer2_0_bn2_bias, metadata["relax.expr.Constant"][12], metadata["relax.expr.Constant"][13], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv29: R.Tensor((1, 128, 28, 28), dtype="float32") = lv28[0]
            lv30: R.Tensor((1, 128, 28, 28), dtype="float32") = R.nn.conv2d(lv22, p_layer2_0_downsample_0_weight, strides=[2, 2], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv31: R.Tuple(R.Tensor((1, 128, 28, 28), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = R.nn.batch_norm(lv30, p_layer2_0_downsample_1_weight, p_layer2_0_downsample_1_bias, metadata["relax.expr.Constant"][14], metadata["relax.expr.Constant"][15], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv32: R.Tensor((1, 128, 28, 28), dtype="float32") = lv31[0]
            lv33: R.Tensor((1, 128, 28, 28), dtype="float32") = R.add(lv29, lv32)
            lv34: R.Tensor((1, 128, 28, 28), dtype="float32") = R.nn.relu(lv33)
            lv35: R.Tensor((1, 128, 28, 28), dtype="float32") = R.nn.conv2d(lv34, p_layer2_1_conv1_weight, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv36: R.Tuple(R.Tensor((1, 128, 28, 28), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = R.nn.batch_norm(lv35, p_layer2_1_bn1_weight, p_layer2_1_bn1_bias, metadata["relax.expr.Constant"][16], metadata["relax.expr.Constant"][17], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv37: R.Tensor((1, 128, 28, 28), dtype="float32") = lv36[0]
            lv38: R.Tensor((1, 128, 28, 28), dtype="float32") = R.nn.relu(lv37)
            lv39: R.Tensor((1, 128, 28, 28), dtype="float32") = R.nn.conv2d(lv38, p_layer2_1_conv2_weight, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv40: R.Tuple(R.Tensor((1, 128, 28, 28), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = R.nn.batch_norm(lv39, p_layer2_1_bn2_weight, p_layer2_1_bn2_bias, metadata["relax.expr.Constant"][18], metadata["relax.expr.Constant"][19], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv41: R.Tensor((1, 128, 28, 28), dtype="float32") = lv40[0]
            lv42: R.Tensor((1, 128, 28, 28), dtype="float32") = R.add(lv41, lv34)
            lv43: R.Tensor((1, 128, 28, 28), dtype="float32") = R.nn.relu(lv42)
            lv44: R.Tensor((1, 256, 14, 14), dtype="float32") = R.nn.conv2d(lv43, p_layer3_0_conv1_weight, strides=[2, 2], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv45: R.Tuple(R.Tensor((1, 256, 14, 14), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = R.nn.batch_norm(lv44, p_layer3_0_bn1_weight, p_layer3_0_bn1_bias, metadata["relax.expr.Constant"][20], metadata["relax.expr.Constant"][21], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv46: R.Tensor((1, 256, 14, 14), dtype="float32") = lv45[0]
            lv47: R.Tensor((1, 256, 14, 14), dtype="float32") = R.nn.relu(lv46)
            lv48: R.Tensor((1, 256, 14, 14), dtype="float32") = R.nn.conv2d(lv47, p_layer3_0_conv2_weight, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv49: R.Tuple(R.Tensor((1, 256, 14, 14), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = R.nn.batch_norm(lv48, p_layer3_0_bn2_weight, p_layer3_0_bn2_bias, metadata["relax.expr.Constant"][22], metadata["relax.expr.Constant"][23], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv50: R.Tensor((1, 256, 14, 14), dtype="float32") = lv49[0]
            lv51: R.Tensor((1, 256, 14, 14), dtype="float32") = R.nn.conv2d(lv43, p_layer3_0_downsample_0_weight, strides=[2, 2], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv52: R.Tuple(R.Tensor((1, 256, 14, 14), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = R.nn.batch_norm(lv51, p_layer3_0_downsample_1_weight, p_layer3_0_downsample_1_bias, metadata["relax.expr.Constant"][24], metadata["relax.expr.Constant"][25], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv53: R.Tensor((1, 256, 14, 14), dtype="float32") = lv52[0]
            lv54: R.Tensor((1, 256, 14, 14), dtype="float32") = R.add(lv50, lv53)
            lv55: R.Tensor((1, 256, 14, 14), dtype="float32") = R.nn.relu(lv54)
            lv56: R.Tensor((1, 256, 14, 14), dtype="float32") = R.nn.conv2d(lv55, p_layer3_1_conv1_weight, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv57: R.Tuple(R.Tensor((1, 256, 14, 14), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = R.nn.batch_norm(lv56, p_layer3_1_bn1_weight, p_layer3_1_bn1_bias, metadata["relax.expr.Constant"][26], metadata["relax.expr.Constant"][27], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv58: R.Tensor((1, 256, 14, 14), dtype="float32") = lv57[0]
            lv59: R.Tensor((1, 256, 14, 14), dtype="float32") = R.nn.relu(lv58)
            lv60: R.Tensor((1, 256, 14, 14), dtype="float32") = R.nn.conv2d(lv59, p_layer3_1_conv2_weight, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv61: R.Tuple(R.Tensor((1, 256, 14, 14), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = R.nn.batch_norm(lv60, p_layer3_1_bn2_weight, p_layer3_1_bn2_bias, metadata["relax.expr.Constant"][28], metadata["relax.expr.Constant"][29], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv62: R.Tensor((1, 256, 14, 14), dtype="float32") = lv61[0]
            lv63: R.Tensor((1, 256, 14, 14), dtype="float32") = R.add(lv62, lv55)
            lv64: R.Tensor((1, 256, 14, 14), dtype="float32") = R.nn.relu(lv63)
            lv65: R.Tensor((1, 512, 7, 7), dtype="float32") = R.nn.conv2d(lv64, p_layer4_0_conv1_weight, strides=[2, 2], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv66: R.Tuple(R.Tensor((1, 512, 7, 7), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = R.nn.batch_norm(lv65, p_layer4_0_bn1_weight, p_layer4_0_bn1_bias, metadata["relax.expr.Constant"][30], metadata["relax.expr.Constant"][31], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv67: R.Tensor((1, 512, 7, 7), dtype="float32") = lv66[0]
            lv68: R.Tensor((1, 512, 7, 7), dtype="float32") = R.nn.relu(lv67)
            lv69: R.Tensor((1, 512, 7, 7), dtype="float32") = R.nn.conv2d(lv68, p_layer4_0_conv2_weight, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv70: R.Tuple(R.Tensor((1, 512, 7, 7), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = R.nn.batch_norm(lv69, p_layer4_0_bn2_weight, p_layer4_0_bn2_bias, metadata["relax.expr.Constant"][32], metadata["relax.expr.Constant"][33], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv71: R.Tensor((1, 512, 7, 7), dtype="float32") = lv70[0]
            lv72: R.Tensor((1, 512, 7, 7), dtype="float32") = R.nn.conv2d(lv64, p_layer4_0_downsample_0_weight, strides=[2, 2], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv73: R.Tuple(R.Tensor((1, 512, 7, 7), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = R.nn.batch_norm(lv72, p_layer4_0_downsample_1_weight, p_layer4_0_downsample_1_bias, metadata["relax.expr.Constant"][34], metadata["relax.expr.Constant"][35], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv74: R.Tensor((1, 512, 7, 7), dtype="float32") = lv73[0]
            lv75: R.Tensor((1, 512, 7, 7), dtype="float32") = R.add(lv71, lv74)
            lv76: R.Tensor((1, 512, 7, 7), dtype="float32") = R.nn.relu(lv75)
            lv77: R.Tensor((1, 512, 7, 7), dtype="float32") = R.nn.conv2d(lv76, p_layer4_1_conv1_weight, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv78: R.Tuple(R.Tensor((1, 512, 7, 7), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = R.nn.batch_norm(lv77, p_layer4_1_bn1_weight, p_layer4_1_bn1_bias, metadata["relax.expr.Constant"][36], metadata["relax.expr.Constant"][37], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv79: R.Tensor((1, 512, 7, 7), dtype="float32") = lv78[0]
            lv80: R.Tensor((1, 512, 7, 7), dtype="float32") = R.nn.relu(lv79)
            lv81: R.Tensor((1, 512, 7, 7), dtype="float32") = R.nn.conv2d(lv80, p_layer4_1_conv2_weight, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="float32")
            lv82: R.Tuple(R.Tensor((1, 512, 7, 7), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = R.nn.batch_norm(lv81, p_layer4_1_bn2_weight, p_layer4_1_bn2_bias, metadata["relax.expr.Constant"][38], metadata["relax.expr.Constant"][39], axis=1, epsilon=1.0000000000000001e-05, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv83: R.Tensor((1, 512, 7, 7), dtype="float32") = lv82[0]
            lv84: R.Tensor((1, 512, 7, 7), dtype="float32") = R.add(lv83, lv76)
            lv85: R.Tensor((1, 512, 7, 7), dtype="float32") = R.nn.relu(lv84)
            lv86: R.Tensor((1, 512, 1, 1), dtype="float32") = R.nn.adaptive_avg_pool2d(lv85, output_size=[1, 1], layout="NCHW", out_layout="NCHW")
            lv87: R.Tensor((1, 512), dtype="float32") = R.reshape(lv86, R.shape([1, 512]))
            lv88: R.Tensor((512, 1000), dtype="float32") = R.permute_dims(p_fc_weight, axes=None)
            lv89: R.Tensor((1, 1000), dtype="float32") = R.matmul(lv87, lv88, out_dtype="float32")
            lv90: R.Tensor((1, 1000), dtype="float32") = R.add(lv89, p_fc_bias)
            gv: R.Tuple(R.Tensor((1, 1000), dtype="float32")) = (lv90,)
            R.output(gv)
        return gv
