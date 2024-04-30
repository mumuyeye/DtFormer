""" 
-*- coding: utf-8 -*-
    @Time    : 2024/1/21  21:05
    @Author  : AresDrw
    @File    : img_aug.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""

import os
from argparse import ArgumentParser

import mmcv
import torch
from PIL import Image
from mmcv.parallel import collate, scatter
from tqdm import tqdm

from mmseg.apis.inference import LoadImage
from mmseg.datasets.pipelines import Compose
from mmseg.models.utils.visualization import colorize_mask, Cityscapes_palette
# from tools.test import update_legacy_cfg

from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_classes, get_palette
import matplotlib.pyplot as plt

import torch.nn.functional as F
import numpy as np

from scipy import misc


def denorm(img, mean, std):
    return img.mul(std).add(mean) / 255.0


mode = 'rain'

ACDC_data_root = f'/hy-tmp/datasets/01-Final_ACDC_for_train'
ACDC_val_data_dir = os.path.join(ACDC_data_root, mode, 'target/img/val')
ACDC_train_data_dir = os.path.join(ACDC_data_root, mode, 'target/img/train')
ACDC_test_data_dir = os.path.join(ACDC_data_root, mode, 'target/img/test')

Foggy_Zurich_data_root = '/hy-tmp/datasets/02-Final_Foggy_Zurich_for_train'
Foggy_Zurich_train_dir = os.path.join(Foggy_Zurich_data_root, 'target/img/train')
Foggy_Zurich_val_dir = os.path.join(Foggy_Zurich_data_root, 'target/img/val')

Foggy_Driving_data_root = '/hy-tmp/datasets/03-Final_Foggy_Driving_for_train'
Foggy_Driving_val_dir = os.path.join(Foggy_Driving_data_root, 'FD/img')
Foggy_Driving_Dense_val_dir = os.path.join(Foggy_Driving_data_root, 'FDD/img')

Dark_Zurich_data_root = '/hy-tmp/datasets/04-Final_Dark_Zurich_for_train'
Dark_Zurich_val_dir = os.path.join(Dark_Zurich_data_root, 'target/img/val')
Dark_Zurich_test_dir = os.path.join(Dark_Zurich_data_root, 'target/img/test')

test_dir = '/hy-tmp/datasets/test_paper'

# config and checkpoint
base_work_dir = f'/hy-tmp/DAFormer/work_dir/test_model/01-SDAT-Former'
# cfg_path = os.path.join(base_work_dir, 'cs2acdc_ref_uda_FITA_rcs_croppl_a999_daformer_mitb5_s0.py')
cfg_path = '/hy-tmp/DAFormer/configs/_base_/datasets/TTA/acdc_1024x1024_repeat.py'
ckpt_path = os.path.join(base_work_dir, 'model.pth')
out_dir_name = f'/hy-tmp/results/aug_images/tta_rain/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


def get_mean_std(img_norm_cfg, dev='cpu'):
    mean = [
        torch.as_tensor(img_norm_cfg['mean'], device=dev)
        for i in range(len([1]))
    ]
    mean = torch.stack(mean).view(-1, 3, 1, 1)
    std = [
        torch.as_tensor(img_norm_cfg['std'], device=dev)
        for i in range(len([1]))
    ]
    std = torch.stack(std).view(-1, 3, 1, 1)
    return mean, std


def main():
    parser = ArgumentParser()
    parser.add_argument('--img', help='Image file')
    parser.add_argument('--img_dir', help='Image dir',
                        default=ACDC_val_data_dir)
    parser.add_argument('--config', help='Config file',
                        default=cfg_path)
    parser.add_argument('--out_dir', help='out dir', default=out_dir_name)
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    cfg = mmcv.Config.fromfile(args.config)
    # cfg = update_legacy_cfg(cfg)

    if args.img_dir is not None:
        outdir = args.out_dir
        augDir = os.path.join(outdir, 'aug')

        os.makedirs(outdir, exist_ok=True)
        os.makedirs(augDir, exist_ok=True)

        for filename in tqdm(os.listdir(args.img_dir)):
            img = os.path.join(args.img_dir, filename)
            file, extension = os.path.splitext(filename)

            # build the data pipeline
            test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
            test_pipeline = Compose(test_pipeline)
            # prepare data
            data = dict(img=img)
            data = test_pipeline(data)
            data = collate([data], samples_per_gpu=1)

            mean, std = get_mean_std(img_norm_cfg, dev='cpu')

            for i in range(len(data['img'])):
                img_tensor = denorm(data['img'][i], mean, std)
                image = torch.clamp(img_tensor[0].permute(1, 2, 0), 0, 1).cpu().numpy()
                plt.imsave(fname=os.path.join(augDir, f'{file}_{i}_aug.png'),
                           arr=image)


if __name__ == '__main__':
    main()
