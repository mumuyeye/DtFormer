# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add additional backbones

from .mix_transformer import (MixVisionTransformer, mit_b0, mit_b1, mit_b2,
                              mit_b3, mit_b4, mit_b5, mit_b5_multi_input)
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1c, ResNetV1d, ResNetLW
from .resnext import ResNeXt
from .swin import SwinTransformer
from .vit import (ImageEncoderViT, ViT_H, ViT_L, ViT_B,
                  PromptedVisionTransformer, Prompted_ViT_B, Prompted_ViT_L, Prompted_ViT_H)

__all__ = [
    'ResNet',
    'ResNetV1c',
    'ResNetV1d',
    'ResNetLW',
    'ResNeXt',
    'ResNeSt',
    'MixVisionTransformer',
    'SwinTransformer',
    'mit_b0',
    'mit_b1',
    'mit_b2',
    'mit_b3',
    'mit_b4',
    'mit_b5',
    'mit_b5_multi_input',
    'ImageEncoderViT',
    'ViT_H',
    'ViT_L',
    'ViT_B',
    'Prompted_ViT_B',
    'Prompted_ViT_L',
    'Prompted_ViT_H'
]
