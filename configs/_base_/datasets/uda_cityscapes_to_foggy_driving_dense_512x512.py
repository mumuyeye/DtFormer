# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# dataset settings
all_data_root = '/raid/wzq/datasets'
cityscapes_data_root = f'{all_data_root}/cityscapes'
ACDC_data_root = f'{all_data_root}/01-Final_ACDC_for_train/fog'
foggy_driving_dense_data_root = f'{all_data_root}/03-Final_Foggy_Driving_for_train/FDD'

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
    samples_per_gpu=2,
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
            ann_dir='target/gt/train/labelTrainIds',
            pipeline=acdc_train_pipeline)),
    val=dict(
        type='FoggyDrivingCoarseDataset', # FDD
        data_root=foggy_driving_dense_data_root,
        img_dir='img',
        ann_dir='gt/labelTrainIds',
        pipeline=test_pipeline),
    test=dict(
        type='FoggyDrivingCoarseDataset',
        data_root=foggy_driving_dense_data_root,
        img_dir='img',
        ann_dir='gt/labelTrainIds',
        pipeline=test_pipeline))
