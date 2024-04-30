# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

_base_ = [
    '../_base_/default_runtime.py',
    # '../_base_/models/daformer_sepaspp_mitb5.py',
    '../_base_/models/deeplabv2_r101-d8.py',
    '../_base_/datasets/acdc_half_512x512.py',
    '../_base_/uda/compare/no_uda.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py'
]
# Random Seed
seed = 0
# Optimizer Hyperparameters
optimizer_config = None
uda = dict(type='NoUDA')
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
n_gpus = 2
runner = dict(type='IterBasedRunner', max_iters=40000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=2)
evaluation = dict(interval=4000, metric='mIoU')
# Meta Information for Result Analysis
name = 'fine_tune_SAM'
exp = 'basic'
name_dataset = 'cityscapes'
name_architecture = 'SAM_VIT_B'
name_encoder = 'ViT_B'
name_decoder = 'MaskDecoder'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
