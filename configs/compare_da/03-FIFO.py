# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/deeplabv2_r101-d8.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/uda_cityscapes_DBF_to_acdc_1024x512.py',
    # Basic UDA Self-Training
    '../_base_/uda/compare/FIFO.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]
# Random Seed
seed = 0
# Optimizer Hyperparameters
optimizer_config = None
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
name = 'da_fifo'
exp = 'basic'
name_dataset = ''
name_architecture = 'deeplabv2'
name_encoder = 'resnet101'
name_decoder = 'deeplabv2_head'
name_uda = 'advent'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
