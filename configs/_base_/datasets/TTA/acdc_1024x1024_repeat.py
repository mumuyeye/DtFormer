# dataset settings
dataset_type = 'ACDCDataset'
all_data_root = '/hy-tmp/datasets'
ACDC_data_root = f'{all_data_root}/01-Final_ACDC_for_train/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (1024, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1920, 1080), ratio_range=(0.5, 2.0)),
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
        img_scale=(1920 // 2, 1080 // 2),
        img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=500,
        dataset=dict(
            type=dataset_type,
            data_root=ACDC_data_root,
            img_dir='leftImg8bit/train',
            ann_dir='gtFine/train',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=ACDC_data_root,
        img_dir='fog/target/img/train',
        ann_dir='fog/target/gt/labelTrainIds/train',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=ACDC_data_root,
        img_dir='fog/target/img/train',
        ann_dir='fog/target/gt/labelTrainIds/train',
        pipeline=test_pipeline),
    test1=dict(
        type=dataset_type,
        data_root=ACDC_data_root,
        img_dir='snow/target/img/train',
        ann_dir='snow/target/gt/labelTrainIds/train',
        pipeline=test_pipeline),
    test2=dict(
        type=dataset_type,
        data_root=ACDC_data_root,
        img_dir='rain/target/img/train',
        ann_dir='rain/target/gt/labelTrainIds/train',
        pipeline=test_pipeline),
    test3=dict(
        type=dataset_type,
        data_root=ACDC_data_root,
        img_dir='night/target/img/train',
        ann_dir='night/target/gt/labelTrainIds/train',
        pipeline=test_pipeline))
