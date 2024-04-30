""" 
-*- coding: utf-8 -*-
    @Time    : 2023/1/19  16:52
    @Author  : AresDrw
    @File    : foggy_driving.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""
from .builder import DATASETS
from .cityscapes import CityscapesDataset


@DATASETS.register_module()
class FoggyDrivingFineDataset(CityscapesDataset):
    def __init__(self, **kwargs):
        super(FoggyDrivingFineDataset, self).__init__(
            img_suffix='_leftImg8bit.png',
            seg_map_suffix='_gtFine_labelTrainIds.png',
            **kwargs)


@DATASETS.register_module()
class FoggyDrivingCoarseDataset(CityscapesDataset):
    def __init__(self, **kwargs):
        super(FoggyDrivingCoarseDataset, self).__init__(
            img_suffix='_leftImg8bit.png',
            seg_map_suffix='_gtCoarse_labelTrainIds.png',
            **kwargs)
