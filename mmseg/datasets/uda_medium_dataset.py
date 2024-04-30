# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import json
import os.path as osp

import mmcv
import numpy as np
import torch

from . import CityscapesDataset
from .builder import DATASETS
from .uda_dataset import UDADataset


@DATASETS.register_module()
class UDAMediumDataset(UDADataset):
    def __init__(self, source, intermediate, target, cfg):
        super().__init__(source, target, cfg)
        self.intermediate = intermediate
        self.intermediate.img_infos = sorted(self.intermediate.img_infos, key=lambda x: x['filename'])
        self.target.img_infos = sorted(self.target.img_infos, key=lambda x: x['filename'])

    def get_rare_class_sample(self):
        """
            TODO:
                This function should be rewritten to adapt the weather condition of target domain
        """
        # 1. choose the data item of source(training) samples
        c = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
        f1 = np.random.choice(self.samples_with_class[c])
        i1 = self.file_to_idx[f1]
        s1 = self.source[i1]
        if self.rcs_min_crop_ratio > 0:
            for j in range(10):
                n_class = torch.sum(s1['gt_semantic_seg'].data == c)
                if n_class > self.rcs_min_pixels * self.rcs_min_crop_ratio:
                    break
                s1 = self.source[i1]

        # 2. choose the data item of target domain randomly
        i3 = np.random.choice(range(len(self.target)))
        s3 = self.target[i3]

        # 3. choose the data item of medium "according to the target item"
        # i2 = i3
        i2 = np.random.choice(range(len(self.intermediate)))
        s2 = self.intermediate[i2]

        return {
            'img': s1['img'],
            'img_metas': s1['img_metas'],
            'gt_semantic_seg': s1['gt_semantic_seg'],
            'imd_img': s2['img'],
            'imd_img_metas': s2['img_metas'],
            'target_img': s3['img'],
            'target_img_metas': s3['img_metas']
        }

    def __getitem__(self, idx):
        if self.rcs_enabled:
            return self.get_rare_class_sample()
        else:
            s1 = self.source[idx // len(self.target)]
            s2 = self.intermediate[idx // len(self.target)]
            s3 = self.target[idx % len(self.target)]
            return {
                'img': s1['img'],
                'img_metas': s1['img_metas'],
                'gt_semantic_seg': s1['gt_semantic_seg'],
                'imd_img': s2['img'],
                'imd_img_metas': s2['img_metas'],
                'gt_imd_semantic_seg': s2['gt_semantic_seg'],
                'target_img': s3['img'],
                'target_img_metas': s3['img_metas']
            }

    def __len__(self):
        return len(self.source) * len(self.target)
