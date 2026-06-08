"""Reference SNN architectures using snn_compiler.

每个工厂函数支持 ``fused=True/False``：
- False: 朴素 nn.Module 网络（Conv→BN→Neuron 三步分离），框架的"baseline"对照
- True : 构造完成后立即 .eval().fuse() / fuse_snn_model，得到融合版

所有模型的 forward 接 ``x_seq: [T, B, 3, H, W]`` 返回 ``[T, B, num_classes]``。
"""
from .vgg import (
    VGGSNN, vgg11_snn, vgg13_snn, vgg16_snn, vgg19_snn,
    VGG11_CFG, VGG13_CFG, VGG16_CFG, VGG19_CFG,
)
from .resnet import (
    BasicBlockSNN, ResNetSNN, resnet18_snn, resnet34_snn,
)
from .sew_resnet import (
    SEWBasicBlockSNN, SEWResNetSNN, sew_resnet18_snn, sew_resnet34_snn,
)
from .mobilenet import (
    InvertedResidualSNN, MobileNetV2SNN, mobilenet_v2_snn, MBV2_CFG,
)

__all__ = [
    "VGGSNN", "vgg11_snn", "vgg13_snn", "vgg16_snn", "vgg19_snn",
    "VGG11_CFG", "VGG13_CFG", "VGG16_CFG", "VGG19_CFG",
    "BasicBlockSNN", "ResNetSNN", "resnet18_snn", "resnet34_snn",
    "SEWBasicBlockSNN", "SEWResNetSNN", "sew_resnet18_snn", "sew_resnet34_snn",
    "InvertedResidualSNN", "MobileNetV2SNN", "mobilenet_v2_snn", "MBV2_CFG",
]
