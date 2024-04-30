""" 
-*- coding: utf-8 -*-
    @Time    : 2023/3/3  13:56
    @Author  : AresDrw
    @File    : adv_FIFO_optimizer.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""
from mmcv.runner import OPTIMIZERS, build_optimizer
from mmcv.runner import DefaultOptimizerConstructor
from torch.optim import Adamax
import torch


@OPTIMIZERS.register_module()
class ADVFifoOptimizer(object):
    def __init__(self):
        self.opt_model = build_optimizer(model, cfg)
        self.opt_fpf = torch.optim.Adamax([p for p in params['model'] if p.requires_grad == True], lr=params['lr_model'])
