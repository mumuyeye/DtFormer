# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# dataset settings
mode = 'snow'
all_data_root = '/hy-tmp/datasets'
cityscapes_data_root = f'{all_data_root}/cityscapes'
ACDC_data_root = f'{all_data_root}/01-Final_ACDC_for_train/{mode}'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (1024, 1024)
cityscapes_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in hrda.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
acdc_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(1920, 1080)),
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
        img_scale=(1920, 1080),
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
        type='UDADataset',
        source=dict(
            type='CityscapesDataset',
            data_root=cityscapes_data_root,
            img_dir='leftImg8bit/train',
            ann_dir='gtFine/train',
            pipeline=cityscapes_train_pipeline),
        target=dict(
            type='ACDCDataset',
            data_root=ACDC_data_root,
            img_dir='target/img/train',
            ann_dir='target/gt/labelTrainIds/train',
            pipeline=acdc_train_pipeline)),
    val=dict(
        type='ACDCDataset',
        data_root=ACDC_data_root,
        img_dir='target/img/val',
        ann_dir='target/gt/labelTrainIds/val',
        pipeline=test_pipeline),
    test=dict(
        type='ACDCDataset',
        data_root=ACDC_data_root,
        img_dir='target/img/val',
        ann_dir='target/gt/labelTrainIds/val',
        pipeline=test_pipeline))
