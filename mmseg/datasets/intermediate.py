""" 
-*- coding: utf-8 -*-
    @Time    : 2023/1/30  14:07
    @Author  : AresDrw
    @File    : intermediate.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""
from .custom import CustomDataset
from .builder import DATASETS
from .cityscapes import CityscapesDataset


@DATASETS.register_module()
class IntermediateDataset(CustomDataset):
    CLASSES = CityscapesDataset.CLASSES
    PALETTE = CityscapesDataset.PALETTE

    def __init__(self, **kwargs):
        super(IntermediateDataset, self).__init__(
            img_suffix='_rgb_ref_anon.png',
            seg_map_suffix='_rgb_ref_anon_labelTrainIds.png',
            **kwargs)
