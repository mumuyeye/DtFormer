""" 
-*- coding: utf-8 -*-
    @Time    : 2023/2/11  18:18
    @Author  : AresDrw
    @File    : CuDAFormer_milestone.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""
# Baseline UDA
uda = dict(
    type='CuDAFormerMilestone',
    alpha=0.99,
    pseudo_threshold=0.968,
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
    pseudo_ref_weight_ignore_top=150,
    use_ref_label=True,
    mix='class',
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    debug_img_interval=1000,
    milestone_for_imd_pseudo_label=12500
)

use_ddp_wrapper = True
