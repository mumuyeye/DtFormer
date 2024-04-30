""" 
-*- coding: utf-8 -*-
    @Time    : 2023/10/18  23:12
    @Author  : AresDrw
    @File    : cdl.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""

from ..builder import DATASETS
from ..custom import CustomDataset


@DATASETS.register_module()
class CDLDataset(CustomDataset):
    CLASSES = ('soya', 'corn')

    PALETTE = [[38, 112, 0], [255, 212, 0]]

    def __init__(self, **kwargs):
        super(CDLDataset, self).__init__(
            img_suffix='.tif',
            seg_map_suffix='_labelTrainId.png',
            **kwargs)
        self.valid_mask_size = [256, 256]


@DATASETS.register_module()
class CDLMultiBandDataset(CDLDataset):
    def __init__(self, **kwargs):
        super(CDLMultiBandDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_labelTrainId.png',
            **kwargs)
