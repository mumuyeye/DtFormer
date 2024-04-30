""" 
-*- coding: utf-8 -*-
    @Time    : 2023/2/11  18:18
    @Author  : AresDrw
    @File    : CuDAFormer_milestone.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""
# Baseline UDA
uda = dict(
    type='CuDAFormerEnhancedBySAM',
    alpha=0.99,
    pseudo_threshold=0.968,
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=150,
    pseudo_ref_weight_ignore_top=150,
    use_ref_label=True,
    mix='class',
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    debug_img_interval=50,
    milestone_for_imd_pseudo_label=0,
    SAM=dict(
        sam_model=dict(model_type='vit_h',
                       checkpoint='/raid/wzq/code/0-experiment-platform/pretrained/SAM/original/sam_vit_h_4b8939.pth',
                       points_per_side=32,
                       pred_iou_thresh=0.8,
                       stability_score_thresh=0.8,
                       crop_n_layers=1,
                       crop_n_points_downscale_factor=2,
                       min_mask_region_area=50),
        device='cuda:0',
        num_cls=19,
        cls_area_threshold=1e4,
        iou_conf_threshold=0.8)
)

use_ddp_wrapper = True
