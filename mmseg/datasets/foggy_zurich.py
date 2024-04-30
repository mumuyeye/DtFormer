""" 
-*- coding: utf-8 -*-
    @Time    : 2023/1/19  16:18
    @Author  : AresDrw
    @File    : foggy_zurich.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""
from .builder import DATASETS
from .cityscapes import CityscapesDataset


@DATASETS.register_module()
class FoggyZurichDataset(CityscapesDataset):
    def __init__(self, **kwargs):
        super(FoggyZurichDataset, self).__init__(
            img_suffix='_rgb_anon.png',
            seg_map_suffix='_gt_labelTrainIds.png',
            **kwargs)


@DATASETS.register_module()
class FoggyZurichLightDataset(CityscapesDataset):
    def __init__(self, **kwargs):
        super(FoggyZurichLightDataset, self).__init__(
            img_suffix='_rgb_ref_anon.png',
            seg_map_suffix='_gt_labelTrainIds.png',
            **kwargs
        )
