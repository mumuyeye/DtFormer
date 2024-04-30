# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# dataset settings
mode = 'fog'
cityscapes_data_root = '/hy-tmp/datasets/cityscapes'
ACDC_data_root = f'/hy-tmp/datasets/01-Final_ACDC_for_train/{mode}'
ACDC_style_root = '/hy-tmp/datasets/StyleTransferACDC'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
cityscapes_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 512)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in hrda.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

acdc_ref_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(960, 540)),  # original 1920x1080
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in hrda.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]

acdc_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(960, 540)),  # original 1920x1080
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in hrda.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(960, 540),  # original 1920x1080
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='UDAMediumDataset',
        source=dict(
            type='CityscapesDataset',
            data_root=cityscapes_data_root,
            img_dir='leftImg8bit/train',
            ann_dir='gtFine/train',
            pipeline=cityscapes_train_pipeline),
        intermediate=dict(
            type='ACDCStyleDataset',
            data_root=ACDC_style_root,
            img_dir=f'cs2acdc{mode}',
            ann_dir='gt/labelTrainIds',
            pipeline=acdc_ref_train_pipeline),
        target=dict(
            type='ACDCDataset',
            data_root=f'{ACDC_data_root}/target',
            img_dir='img/train',
            ann_dir='gt/labelTrainIds/train',
            pipeline=acdc_train_pipeline)),
    val=dict(
        type='ACDCDataset',
        data_root=f'{ACDC_data_root}/target',
        img_dir='img/val',
        ann_dir='gt/labelTrainIds/val',
        pipeline=test_pipeline),
    test=dict(
        type='ACDCDataset',
        data_root=f'{ACDC_data_root}/target',
        img_dir='img/train',
        ann_dir='gt/labelTrainIds/train',
        pipeline=test_pipeline))
