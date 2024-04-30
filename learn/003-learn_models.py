""" 
-*- coding: utf-8 -*-
    @Time    : 2023/2/21  9:07
    @Author  : AresDrw
    @File    : 003-learn_models.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""
import torch
from mmcv import Config

from mmseg.apis import init_segmentor
from mmseg.core import get_classes, get_palette
from mmseg.models.builder import build_backbone
from mmseg.models.builder import build_head
from mmseg.models.builder import build_neck
from mmseg.models.builder import build_segmentor

if __name__ == "__main__":
    cfg_file = '/configs/_base_/models/daswin_conv1_swin_large_patch4_window12_384_22k.py'
    cfg = Config.fromfile(cfg_file)
    segmentor = build_segmentor(cfg['model']).cuda()
    checkpoint = '/raid/wzq/code/0-experiment-platform/pretrained/SwinTransformer' \
                 '/mmseg_swin_large_patch4_window12_384_22k' \ 
                 '.pth'
    # x = torch.randn(1, 3, 512, 512).cuda()
    # out = sam.encode_decode(x, img_metas=None)
    model = init_segmentor(
        cfg,
        checkpoint,
        device=0,
        classes=get_classes('cityscapes'),
        palette=get_palette('cityscapes'),
        revise_checkpoint=[(r'^module\.', ''), ('model.', '')])
    print('done')
