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
# from tools.test import update_legacy_cfg

from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_classes, get_palette
import matplotlib.pyplot as plt

import torch.nn.functional as F
import numpy as np

from scipy import misc

mode = 'snow'

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
kaifeng_dir = f'/hy-tmp/datasets/KaiFengNight'


method = 'milestone_12500_fz_imd_600'
zhengzhou_dir = f'/hy-tmp/datasets/Zhengzhou/{mode}'

input_dir = ACDC_val_data_dir
# base_work_dir = f'/hy-tmp/DAFormer/work_dir/test_model/01-SDAT-Former'
cfg_path = '/hy-tmp/DAFormer/configs/daformer/cs_adverse_uda_daformer_mitb5_s0.py'
ckpt_path = f'/hy-tmp/DAFormer/pretrained/{method}/model.pth'

# if 'KaiFeng' in input_dir:
#     out_dir_name = f'/hy-tmp/{method}/kaifeng'
# else:
#     out_dir_name = f'/hy-tmp/{method}/zhengzhou/{mode}'

data = 'acdc_val'
out_dir_name = f'/hy-tmp/result_add/{method}/{data}/{mode}'


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
                        default=input_dir)
    parser.add_argument('--config', help='Config file',
                        default=cfg_path)
    parser.add_argument('--checkpoint', help='Checkpoint file',
                        default=ckpt_path)
    parser.add_argument('--out_dir', help='out dir', default=out_dir_name)
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
    # cfg = update_legacy_cfg(cfg)
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
        outdir = args.out_dir
        # trainLabelIdDir = os.path.join(outdir, 'labelTrainIds')
        # colorImageDir = os.path.join(outdir, 'colorImage')
        # effectImageDir = os.path.join(outdir, 'effectImage')
        # entropyImageDir = os.path.join(outdir, 'entropyImage')
        # attnImageDir = os.path.join(outdir, 'attenImage')

        os.makedirs(outdir, exist_ok=True)
        # os.makedirs(trainLabelIdDir, exist_ok=True)
        # os.makedirs(colorImageDir, exist_ok=True)
        # os.makedirs(effectImageDir, exist_ok=True)
        # os.makedirs(entropyImageDir, exist_ok=True)

        for filename in tqdm(os.listdir(args.img_dir)):
            img = os.path.join(args.img_dir, filename)
            file, extension = os.path.splitext(filename)

            # generate the result
            # result = inference_segmentor(model, img)

            cfg = model.cfg
            device = next(model.parameters()).device  # model device
            # build the data pipeline
            test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
            test_pipeline = Compose(test_pipeline)
            # prepare data
            data = dict(img=img)
            data = test_pipeline(data)
            data = collate([data], samples_per_gpu=1)
            if next(model.parameters()).is_cuda:
                # scatter to specified GPU
                data = scatter(data, [device])[0]
            else:
                data['img_metas'] = [i.data[0] for i in data['img_metas']]

            # forward the model
            with torch.no_grad():
                new_data = F.interpolate(data['img'][0], size=(512, 1024))
                cls_score = model.encode_decode(img=new_data, img_metas=data['img_metas'][0])
                cls_prob = F.interpolate(torch.softmax(cls_score, dim=1), size=(512, 1024))
                # entropy_map = calc_entropy(cls_prob).squeeze(dim=0).cpu().detach().numpy()
                _, result = torch.max(cls_prob, dim=1)

            # # save the entropy map
            # plt.imsave(fname=os.path.join(entropyImageDir, f'{file}_entropy.png'),
            #            arr=entropy_map, cmap='viridis')

            # # save the attention map
            # plt.imsave(fname=os.path.join(attnImageDir, f'{file}_atten.png'),
            #            arr=atten_map, cmap='gray')

            # # save the labelTrainIds:
            # labelTrainId = Image.fromarray(result[0].cpu().numpy().astype('uint8')).convert('P')
            # labelTrainId.save(os.path.join(trainLabelIdDir, f'{file}_gt_labelTrainIds.png'))

            # save the colorImages:
            color_img = colorize_mask(result[0].cpu().numpy(), palette=Cityscapes_palette)  # 'PIL'
            color_img.save(os.path.join(outdir, f'{file}_color.png'))

            # show the results
            # pred_file = os.path.join(effectImageDir, f'{file}_pred_effect{extension}')
            # assert pred_file != args.img
            # model.show_result(
            #     img,
            #     result.cpu().numpy(),
            #     palette=get_palette(args.palette),
            #     out_file=pred_file,
            #     show=False,
            #     opacity=args.opacity)


if __name__ == '__main__':
    main()
