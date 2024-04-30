""" 
-*- coding: utf-8 -*-
    @Time    : 2023/3/3  10:23
    @Author  : AresDrw
    @File    : fifo.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""
uda = dict(
    type='FIFO',
    debug_img_interval=1000,
    fog_pass_filter=dict(in_size_0=2080, in_size_1=32896),  # 64, 256
    lr_fpf_params=dict(lr=1e-4, power=0.9, min=0))
