# model settings
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='SAMEncoderDecoder',
    backbone=dict(type='ViT_B'),
    neck=dict(type='PromptEncoder'),
    decode_head=dict(
        type='MaskDecoder',
        in_channels=256,
        channels=64,
        num_classes=19,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', stride=(1024, 1024), crop_size=(1024, 1024)))
