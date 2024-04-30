""" 
-*- coding: utf-8 -*-
    @Time    : 2023/5/25  12:17
    @Author  : AresDrw
    @File    : enhance_pseudo_label_by_iou.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""
import os

import torch
import cv2
import numpy as np
from PIL import Image
import torch.nn.functional as F

from mmseg.models.utils.visualization import colorize_mask, Cityscapes_palette

debug_dir = '/raid/wzq/code/0-experiment-platform/test_image/test_enhance_sam_pseudo_label/foggyzurich01/debug'


def replace_values(dst, src):
    mask = (dst == -1)
    src = src.long()
    dst[mask] = src[mask]
    return dst


def fill_nonzero_values(src_tensor, tgt_tensor):
    """
    填充张量tgt_tensor中与src_tensor相同位置的值。
    使用src_tensor中不为0的值填充tgt_tensor中相同位置的值。
    :param src_tensor: 源张量: [1, 1, 1024, 2048]
    :param tgt_tensor: 目标张量: [1, 1024, 2048]
    :return: 填充后的目标张量tgt_tensor
    """
    mask = torch.nonzero(src_tensor)
    values = src_tensor[mask[:, 0], mask[:, 1], mask[:, 2]].long()
    tgt_tensor[mask[:, 0], mask[:, 1], mask[:, 2]] = values
    return tgt_tensor


def merge_pseudo_label(pl_teacher, sam_segments,
                       num_cls=19, pl_threshold=0.75,
                       cls_threshold=10000):
    """
    This function will use the sam_segments
    to enhance the pseudo label generated from teacher

    :param pl_teacher:
        pseudo_label generated from teacher
        Tensor: [B, H, W], cls_id
    :param sam_segments:
        segments from 'Everything mode' of SAM_masks
        List[Tensor[B, H, W]] bool
    :param num_cls:
        number of classes, e.g., 19
    :param pl_threshold:
        the threshold of judgements, e.g., 0.9
    :return: enhanced_pl
        Tensor: [B, H, W], cls_id
    """
    # slice the pl_teacher
    cls_bool_slices = []
    for cls_id in range(num_cls):
        cls_mask = (pl_teacher == cls_id).bool()
        cls_bool_slices.append(cls_mask)

    # initialize the enhanced_pseudo_label
    cls_sam_masks = [[] for _ in range(19)]
    enhanced_pl = torch.ones((1, 1080, 1920), dtype=torch.long) * -1

    # 将B中的每个二值掩膜与C中的每个掩膜计算Recall，若Recall大于T则合并并标记
    for cls_id, cls_bool_slice in enumerate(cls_bool_slices):
        if cls_bool_slice.sum() > cls_threshold:  # get rid of no-showed classes
            for j, sam_segment in enumerate(sam_segments):
                sam_segment = sam_segment / 255  # preprocess the sam from 255 into 1
                iou_mask = cls_bool_slice * sam_segment
                intersect = (cls_bool_slice * sam_segment).sum()
                recall_pl = intersect / cls_bool_slice.sum()
                recall_sam = intersect / sam_segment.sum()
                if recall_sam > pl_threshold or recall_pl > pl_threshold:
                    # debug
                    iou_mask_save = Image.fromarray(iou_mask[0].cpu().numpy().astype('uint8') * 255).convert('P')
                    iou_mask_save.save(os.path.join(debug_dir, 'intersect', f'cls_{cls_id}_sam_{j}_rsam_{recall_sam}_rpl_{recall_pl}.png'))
                    sam_segment_save = Image.fromarray(sam_segment[0].cpu().numpy().astype('uint8') * 255).convert('P')
                    sam_segment_save.save(
                        os.path.join(debug_dir, 'sam_segments', f'cls_{cls_id}_sam_{j}_rsam_{recall_sam}_rpl_{recall_pl}.png'))
                    cls_sam_masks[cls_id].append(sam_segment)

    # merge the semantic mask in cls_sam_masks
    for cls_id, d in enumerate(cls_sam_masks):
        if len(d) > 0:
            dp = torch.stack(d)
            d_sum = dp.sum(dim=0)
            d_sum[d_sum >= 1] = 1
            cls_sam_masks[cls_id] = d_sum
            # # debug cls_sam_mask_save = Image.fromarray(cls_sam_masks[cls_id][0].cpu().numpy().astype('uint8') *
            # 255).convert('P') cls_sam_mask_save.save( os.path.join(debug_dir, 'merge_sam_masks',
            # f'cls_{cls_id}_sam.png'))

    # merge the cls_sam_masks -> enhanced_pl
    for cls_id, d in enumerate(cls_sam_masks):
        if len(d) > 0:
            if cls_id == 0:
                enhanced_pl = fill_nonzero_values(src_tensor=d, tgt_tensor=enhanced_pl)
                enhanced_pl[enhanced_pl == 1] = cls_id
            else:
                d[d == 1] = cls_id
                enhanced_pl = fill_nonzero_values(src_tensor=d, tgt_tensor=enhanced_pl)

    # fill the last -1 -> pl_teacher
    enhanced_pl = replace_values(enhanced_pl, pl_teacher)
    return enhanced_pl


if __name__ == "__main__":
    pl_teacher = cv2.imread('/raid/wzq/code/0-experiment-platform/test_image/test_enhance_sam_pseudo_label'
                            '/foggyzurich01/out_pred/labelTrainIds/Foggy_Zurich_001_rgb_anon_gt_labelTrainIds.png', cv2.IMREAD_GRAYSCALE)
    pl_teacher = F.interpolate(torch.from_numpy(pl_teacher).unsqueeze(dim=0).unsqueeze(dim=0), size=(1080, 1920)).squeeze(dim=0)
    sam_mask_dir = '/raid/wzq/code/0-experiment-platform/test_image/test_enhance_sam_pseudo_label/foggyzurich01' \
                   '/SAM_masks'
    masks = []
    for mask in os.listdir(sam_mask_dir):
        sam_mask = torch.from_numpy(cv2.imread(os.path.join(sam_mask_dir, mask)))[:, :, 0].unsqueeze(dim=0)
        masks.append(sam_mask)

    pl_enhanced = merge_pseudo_label(pl_teacher, masks)

    color_img = colorize_mask(pl_enhanced[0].cpu().numpy(), palette=Cityscapes_palette)  # 'PIL'
    color_img.save(os.path.join('/raid/wzq/code/0-experiment-platform/test_image/test_enhance_sam_pseudo_label'
                                '/foggyzurich01/enhanced_mask',
                                'enahnced_color.png'))
