""" 
-*- coding: utf-8 -*-
    @Time    : 2023/2/5  20:00
    @Author  : AresDrw
    @File    : 001-learn_dataset.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-

   导入所需要的包：
        注意，一旦被注册进入了openmmlab的registry中
        就不能再通过原有的构造函数进行创建实例了
        ——只能使用带有的build函数进行构建
"""
from mmseg.datasets.builder import build_dataset
from mmseg.datasets.builder import build_dataloader
from mmcv.utils.config import Config

if __name__ == "__main__":

    # 1.普通的数据集：Cityscapes
    # cfg_file = '/hy-tmp/01-DAFormer/configs/_base_/datasets/cdl_rgb_half_512x512.py'
    # 2. UDA数据集：cityscapes->ACDC
    cfg_file = '/hy-tmp/exp-platform/configs/_base_/datasets/uda_cityscapes_to_intermediate_acdc_512x512.py'

    data_cfg = Config.fromfile(cfg_file)
    cs_dataset = build_dataset(data_cfg.data.train)  # must have 'type'
    cs_dataloader = build_dataloader(dataset=cs_dataset,
                                     samples_per_gpu=2,
                                     workers_per_gpu=1)

    for i, data in enumerate(cs_dataloader):
        if i == 0:
            print('data.img_metas:', data['img_metas'].data[0][0]['filename'])
            print('gt', data['gt_semantic_seg'].data[0])
            print('gt.size', data['gt_semantic_seg'].data[0].size())
            print('data.target_img_metas:', data['img_metas'].data[0][0]['filename'])
            break
