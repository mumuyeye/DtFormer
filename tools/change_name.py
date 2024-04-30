""" 
-*- coding: utf-8 -*-
    @Time    : 2023/10/19  15:55
    @Author  : AresDrw
    @File    : change_name.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""
import os
from PIL import Image
import numpy as np
import os.path as osp

root_dir = '/hy-tmp/datasets/StyleTransferACDC'

for dir in os.listdir(root_dir):
    for file in os.listdir(osp.join(root_dir, dir)):
        if file.endswith('_style.png'):
            continue
        else:
            new_name = f'{file[4:-4]}_style.png'
            os.rename(src=osp.join(root_dir, dir, file),
                      dst=osp.join(root_dir, dir, new_name))

