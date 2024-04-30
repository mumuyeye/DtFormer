# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Half image resolution

# dataset settings
mode = 'fog'
all_data_root = '/hy-tmp/datasets/01-Final_ACDC_for_train'
acdc_data_root = f'{all_data_root}/{mode}'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 1024)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
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
        type='ACDCDataset',
        data_root=acdc_data_root,
        img_dir='target/img/train',
        ann_dir='target/gt/labelTrainIds/train',
        pipeline=train_pipeline),
    val=dict(
        type='ACDCDataset',
        data_root=acdc_data_root,
        img_dir='target/img/val',
        ann_dir='target/gt/labelTrainIds/val',
        pipeline=test_pipeline),
    test=dict(
        type='ACDCDataset',
        data_root=acdc_data_root,
        img_dir='target/img/val',
        ann_dir='target/gt/labelTrainIds/val',
        pipeline=test_pipeline))
