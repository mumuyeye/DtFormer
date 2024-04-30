""" 
-*- coding: utf-8 -*-
    @Time    : 2022/12/29  15:53
    @Author  : AresDrw
    @File    : colorImage2label.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""
import os
from collections import namedtuple
from PIL import Image
import numpy as np
from tqdm import tqdm
import shutil

Cls = namedtuple('cls', ['name', 'id', 'color'])
Clss = [
    Cls('road', 0, (128, 64, 128)),
    Cls('sidewalk', 1, (244, 35, 232)),
    Cls('building', 2, (70, 70, 70)),
    Cls('wall', 3, (102, 102, 156)),
    Cls('fence', 4, (190, 153, 153)),
    Cls('pole', 5, (153, 153, 153)),
    Cls('traffic light', 6, (250, 170, 30)),
    Cls('traffic sign', 7, (220, 220, 0)),
    Cls('vegetation', 8, (107, 142, 35)),
    Cls('terrain', 9, (152, 251, 152)),
    Cls('sky', 10, (70, 130, 180)),
    Cls('person', 11, (220, 20, 60)),
    Cls('rider', 12, (255, 0, 0)),
    Cls('car', 13, (0, 0, 142)),
    Cls('truck', 14, (0, 0, 70)),
    Cls('bus', 15, (0, 60, 100)),
    Cls('train', 16, (0, 80, 100)),
    Cls('motorcycle', 17, (0, 0, 230)),
    Cls('bicycle', 18, (119, 11, 32))
]


def colorImage2label(img_dir, out_dir, img_suffix='_rgb_ref_anno_color.png'):
    for filename in tqdm(os.listdir(img_dir)):
        colorImage = np.array(Image.open(os.path.join(img_dir, filename)).convert('RGB'))
        labelTrainId = 255 * np.ones((colorImage.shape[0], colorImage.shape[1]), dtype='uint8')
        r = colorImage[:, :, 0]
        g = colorImage[:, :, 1]
        b = colorImage[:, :, 2]

        for cls in Clss:
            cur_cls_r_mask = r == cls.color[0]
            cur_cls_g_mask = g == cls.color[1]
            cur_cls_b_mask = b == cls.color[2]
            cur_cls_mask = cur_cls_r_mask * cur_cls_g_mask * cur_cls_b_mask
            labelTrainId[cur_cls_mask] = cls.id

        # 'GP010478_frame_000167_rgb_ref_anon_color.png'
        filename = filename[:-23]
        labelTrainIdImage = Image.fromarray(labelTrainId).convert('P')
        labelTrainIdImage.save(os.path.join(out_dir, f'{filename}_gt_labelTrainIds.png'))


if __name__ == "__main__":
    # colorImage2label(img_dir=r'D:\data\ACDC_fog_train\ref\gt\colorImage',
    #                  out_dir=r'D:\data\ACDC_fog_train\ref\gt\trainLabelId')
    label_path = r'C:\Users\17138\Desktop\acdcref_pseudo\labelTrainIds\GOPR0475_frame_000256_rgb_anon_labelTrainIds.png'
    label = np.array(Image.open(label_path))
    print('done')

    # img_dir = r'D:\data\ACDC_fog_train\fog\gt\val'
    # for vid in os.listdir(img_dir):
    #     for img in os.listdir(os.path.join(img_dir, vid)):
    #         if 'gt_labelTrainIds' in img:
    #             shutil.copy(os.path.join(img_dir, vid, img), r'D:\data\ACDC_fog_train\fog\gt\trainLabelIds')
    #         elif 'gt_labelColor' in img:
    #             shutil.copy(os.path.join(img_dir, vid, img), r'D:\data\ACDC_fog_train\fog\gt\colorLabel')

    print('done')
