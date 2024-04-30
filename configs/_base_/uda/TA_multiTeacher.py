""" 
-*- coding: utf-8 -*-
    @Time    : 2023/2/23  15:37
    @Author  : AresDrw
    @File    : TA_multiTeacher.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""
uda = dict(
    type='MultiTeacherIMDTGT',
    alpha=0.99,
    pseudo_threshold=0.968,
    pseudo_weight_ignore_top=200,
    pseudo_weight_ignore_bottom=0,
    pseudo_ref_weight_ignore_top=150,
    use_ref_label=True,
    mix='class',
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    debug_img_interval=999,
)

use_ddp_wrapper = True