# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# DAFormer (with context-aware feature fusion) in Tab. 7
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='TransWeatherSegmentor',
    # backbone=dict(type='mit_b5',
    #               pretrained='/home/gis/wzq/code/0-experiment-platform/pretrained/mit_b5.pth'),
    backbone=dict(type='TransWeatherBackbone',
                  init_cfg=dict(type='Pretrained',
                                checkpoint='/home/gis/wzq/code/0-experiment-platform/pretrained/tw_encoder_200e.pth')),
    restore_decode_head=dict(type='TransWeatherDecoder',
                             init_cfg=dict(type='Pretrained',
                                       checkpoint='/home/gis/wzq/code/0-experiment-platform/pretrained/tw_decoder_200e.pth')),
    decode_head=dict(
        type='DAFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(
            embed_dims=256,
            embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            fusion_cfg=dict(
                type='conv',
                kernel_size=1,
                act_cfg=dict(type='ReLU'),
                norm_cfg=norm_cfg),
        ),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))