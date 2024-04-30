# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications: Add valid_mask_size
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from .builder import DATASETS
from .cityscapes import CityscapesDataset


@DATASETS.register_module()
class ACDCDataset(CityscapesDataset):

    def __init__(self, **kwargs):
        super(ACDCDataset, self).__init__(
            img_suffix='_rgb_anon.png',
            seg_map_suffix='_gt_labelTrainIds.png',
            **kwargs)
        self.valid_mask_size = [1080, 1920]


@DATASETS.register_module()
class ACDCRefDataset(CityscapesDataset):
    def __init__(self, **kwargs):
        super(ACDCRefDataset, self).__init__(
            img_suffix='_rgb_ref_anon.png',
            seg_map_suffix='_gt_labelTrainIds.png',
            **kwargs)
        self.valid_mask_size = [1080, 1920]


@DATASETS.register_module()
class ACDCStyleDataset(CityscapesDataset):
    def __init__(self, **kwargs):
        super(ACDCStyleDataset, self).__init__(
            img_suffix='rgb_anon_style.png',
            seg_map_suffix='_gt_labelTrainIds.png',
            **kwargs)
        self.valid_mask_size = [1080, 1920]
