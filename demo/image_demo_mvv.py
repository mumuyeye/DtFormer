# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Config and checkpoint update
# - Saving instead of showing prediction

import os
import mmcv
import numpy
import torch
from mmcv.parallel import collate, scatter
from prettytable import PrettyTable
from tqdm import tqdm
from mmseg.apis.inference import LoadImage
from mmseg.datasets.pipelines import Compose
from mmseg.apis import init_segmentor
from mmseg.core.evaluation import get_classes, get_palette
import torch.nn.functional as F

cityscapes_root = '/hy-tmp/datasets/cs_all'
ACDC_data_root = f'/hy-tmp/datasets/01-Final_ACDC_for_train/fog'
ACDC_train_data_dir = os.path.join(ACDC_data_root, 'target/img/train')
ACDC_val_data_dir = os.path.join(ACDC_data_root, 'target/img/val')
ACDC_test_data_dir = os.path.join(ACDC_data_root, 'target/img/test')

Foggy_Zurich_val_dir = '/hy-tmp/datasets/02-Final_Foggy_Zurich_for_train'

Foggy_Driving_data_root = '/hy-tmp/datasets/03-Final_Foggy_Driving_for_train'
Foggy_Driving_val_dir = os.path.join(Foggy_Driving_data_root, 'FD/img')
Foggy_Driving_Dense_val_dir = os.path.join(Foggy_Driving_data_root, 'FDD/img')

Dark_Zurich_data_root = '/hy-tmp/datasets/04-Final_Dark_Zurich_for_train'
Dark_Zurich_val_dir = os.path.join(Dark_Zurich_data_root, 'target/img/val')
Dark_Zurich_test_dir = os.path.join(Dark_Zurich_data_root, 'target/img/test')


def calc_entropy(prob):
    """
    :param prob: softmax of the score
    :return:
    """
    entropy_map = torch.sum(-prob * torch.log(prob + 1e-7), dim=1)
    return entropy_map


def get_data_name(data_dir):
    domain = None
    if 'ACDC' in data_dir:
        domain = 'ACDC'
    if 'cs' in data_dir:
        domain = 'Cityscapes'
    if 'Foggy_Zurich' in data_dir:
        domain = 'FZ'
    if 'FD/' in data_dir:
        domain = 'FD'
    if 'FDD/' in data_dir:
        domain = 'FDD'
    return domain


def get_single_domain_uncertainty(data_dir, model):
    data_name = get_data_name(data_dir)
    device = 'cuda:0'
    current_domain_ent_l = []
    for filename in tqdm(os.listdir(data_dir)):
        img = os.path.join(data_dir, filename)
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
            entropy_map = calc_entropy(cls_prob).squeeze(dim=0).cpu().detach().numpy()
            current_domain_ent_l.append(entropy_map.mean())

    return [data_name, sum(current_domain_ent_l) / len(current_domain_ent_l)]


def main():
    root = '/hy-tmp/DAFormer'
    domains = [cityscapes_root, ACDC_test_data_dir, Foggy_Zurich_val_dir,
               Foggy_Driving_val_dir, Foggy_Driving_Dense_val_dir]
    domain_names = [get_data_name(data) for data in domains]
    checkpoints = []
    cfg_path = '/hy-tmp/DAFormer/configs/daformer/cs_adverse_uda_daformer_mitb5_s0.py'
    for file in os.listdir(os.path.join(root, 'pretrained')):
        if file.endswith('segformer_model.pth'):
            method_name = os.path.basename(file).replace('_model.pth', '')
            checkpoints.append(dict(name=method_name,
                                    path=os.path.join(root, 'pretrained', file)))

    table = PrettyTable()
    table.title = 'MVV Domain Gap results Info'
    table.field_names = [''] + domain_names

    cfg = mmcv.Config.fromfile(cfg_path)

    out_file = '/hy-tmp/mvv_result.txt'
    with open(out_file, 'w') as f:
        for i, ckpt in enumerate(checkpoints):
            model = init_segmentor(
                cfg,
                ckpt['path'],
                device='cuda:0',
                classes=get_classes('cityscapes'),
                palette=get_palette('cityscapes'),
                revise_checkpoint=[(r'^module\.', ''), ('model.', '')])
            row_info = [ckpt['name']]
            for domain in domains:
                info = get_single_domain_uncertainty(data_dir=domain, model=model)
                row_info.append(info[-1])
            table.add_row(row_info)
            print(table)
        f.write(str(table))


if __name__ == '__main__':
    main()
