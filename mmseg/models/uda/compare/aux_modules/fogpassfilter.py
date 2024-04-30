""" 
-*- coding: utf-8 -*-
    @Time    : 2023/1/10  22:24
    @Author  : AresDrw
    @File    : fogpassfilter.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""
from abc import ABC

import torch.nn as nn
from mmcv.runner import BaseModule

from mmseg.models.utils.basic_blocks import *


class FogPassFilter_conv1(nn.Module, ABC):
    def __init__(self, inputsize):
        super(FogPassFilter_conv1, self).__init__()

        self.hidden = nn.Linear(inputsize, inputsize // 2)
        self.hidden2 = nn.Linear(inputsize // 2, inputsize // 4)
        self.output = nn.Linear(inputsize // 4, 64)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.hidden(x)
        x = self.leakyrelu(x)
        x = self.hidden2(x)
        x = self.leakyrelu(x)
        x = self.output(x)
        return x


class FogPassFilter_res1(nn.Module, ABC):
    def __init__(self, inputsize):
        super(FogPassFilter_res1, self).__init__()
        self.hidden = nn.Linear(inputsize, inputsize // 8)
        self.output = nn.Linear(inputsize // 8, 64)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.hidden(x)
        x = self.leakyrelu(x)
        x = self.output(x)
        return x


class FogPassFilter(BaseModule):
    def __init__(self, input_size_0, input_size_1):
        super(FogPassFilter, self).__init__()
        self.filter0 = FogPassFilter_conv1(inputsize=input_size_0)
        self.filter1 = FogPassFilter_res1(inputsize=input_size_1)

    def forward(self, x, layer):
        if layer == 0:
            return self.filter0(x)
        elif layer == 1:
            return self.filter1(x)
