# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# Medium UDA TransWeather MultiTask
uda = dict(
    type='UDAMediumTransWeatherMulti',
    alpha=0.99,
    pseudo_threshold=0.968,
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
    mix='class',
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    debug_img_interval=1000,
    print_grad_magnitude=False,
    discriminator=dict(type='DISE_Discriminator',
                       in_channels=3),
    domain_classifier=dict(type='DiseDomainClassifier',
                           n_classes=19)
)
use_ddp_wrapper = True
