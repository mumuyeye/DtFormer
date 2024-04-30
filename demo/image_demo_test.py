# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Config and checkpoint update
# - Saving instead of showing prediction

import cycler

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
from tools.test import update_legacy_cfg

from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_classes, get_palette
import matplotlib.pyplot as plt

import torch.nn.functional as F

from scipy import misc


def calc_entropy(prob):
    """
    :param prob: softmax of the score
    :return:
    """
    entropy_map = torch.sum(-prob * torch.log(prob + 1e-7), dim=1)
    return entropy_map


def main():
    parser = ArgumentParser()
    parser.add_argument('--img', help='Image file')
    parser.add_argument('--img_dir', help='Image dir',
                        default='/raid/wzq/datasets/paper/xformer_cum/img_inf')
    parser.add_argument('--config', help='Config file',
                        default='/raid/wzq/code/0-experiment-platform/work_dirs/cudaformer/snow/acdc_snow/valid_test_m_15000/'
                                'cs_adverse_uda_daformer_mitb5_s0.py')
    parser.add_argument('--checkpoint', help='Checkpoint file',
                        default='/raid/wzq/code/0-experiment-platform/work_dirs/cudaformer/snow/acdc_snow/valid_test_m_15000'
                                '/iter_40000.pth')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    cfg = mmcv.Config.fromfile(args.config)
    cfg = update_legacy_cfg(cfg)
    model = init_segmentor(
        cfg,
        args.checkpoint,
        device=args.device,
        classes=get_classes(args.palette),
        palette=get_palette(args.palette),
        revise_checkpoint=[(r'^module\.', ''), ('model.', '')])
    # test a single image

    if args.img is not None:
        result = inference_segmentor(model, args.img)
        # show the results
        file, extension = os.path.splitext(args.img)
        pred_file = f'{file}_pred{extension}'
        assert pred_file != args.img
        model.show_result(
            args.img,
            result,
            palette=get_palette(args.palette),
            out_file=pred_file,
            show=False,
            opacity=args.opacity)
        print('Save prediction to', pred_file)

    if args.img_dir is not None:
        outdir = '/raid/wzq/datasets/paper/xformer_cum/'
        trainLabelIdDir = os.path.join(outdir, 'labelTrainIds')
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(trainLabelIdDir, exist_ok=True)

        for filename in tqdm(os.listdir(args.img_dir)):
            img = os.path.join(args.img_dir, filename)
            result = inference_segmentor(model, img)
            file, extension = os.path.splitext(filename)

            # result : list[numpy(1080, 1920)]
            labelTrainId = Image.fromarray(result[0].astype('uint8')).convert('P')
            labelTrainId.save(os.path.join(trainLabelIdDir, f'{file}_gt_labelTrainIds.png'))


if __name__ == '__main__':
    main()
