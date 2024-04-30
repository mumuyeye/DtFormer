# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/daformer_dahead_transweather_b5.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/uda_cityscapes_to_intermediate_acdc_512x512.py',
    # Basic UDA Self-Training
    '../_base_/uda/uda_medium_trans_weather.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]
# Random Seed
seed = 0
# Modifications to Basic UDA
uda = dict(
    # Increased Alpha
    alpha=0.999,
    # Pseudo-Label Crop
    pseudo_weight_ignore_top=15,
    pseudo_weight_ignore_bottom=120)
data = dict(
    train=dict(
        # Rare Class Sampling
        rare_class_sampling=dict(
            min_pixels=3000, class_temp=0.01, min_crop_ratio=0.5)))
# Optimizer Hyperparameters
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=40000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=40000, max_keep_ckpts=1)
evaluation = dict(interval=4000, metric='mIoU')
# Meta Information for Result Analysis
name = 'cs2acdc_uda_medium_warm_rcs_croppl_a999_transweather_multitask_mitb5_s0.py'
exp = 'basic'
name_dataset = 'cityscapes2acdc'
name_architecture = 'daformer_dahead_transweather_b5'
name_encoder = 'transweather_backbone'
name_decoder = 'daformer_dahead'
name_uda = 'transweather_multitask'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
