import torch
import numpy as np
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F
from torch.utils import data
from tqdm import tqdm
from MulticoreTSNE import MulticoreTSNE as TSNE
from mmcv import Config
from tools.test import update_legacy_cfg

from mmseg.apis import inference_segmentor, init_segmentor

from mmseg.datasets.builder import build_dataset
from mmseg.datasets.builder import build_dataloader
from mmseg.models.builder import build_segmentor
from mmseg.core.evaluation import get_classes, get_palette

from time import time


def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)


device = 0
NUM_CLASSES = 2
input_size = 512
MULTI_LEVEL = True


def data_preparation(d_cfg, m_cfg, checkpoint, mode, n_emb_source, n_emb_target):
    d_cfg = Config.fromfile(d_cfg)
    m_cfg = Config.fromfile(m_cfg)

    source_dataset = build_dataset(d_cfg.data.train.target)
    source_loader = build_dataloader(dataset=source_dataset,
                                     samples_per_gpu=1,
                                     workers_per_gpu=4)

    target_dataset = build_dataset(d_cfg.data.train.target)
    target_loader = build_dataloader(dataset=target_dataset,
                                     samples_per_gpu=1,
                                     workers_per_gpu=4)

    model = init_segmentor(
        m_cfg,
        checkpoint,
        device=device,
        classes=get_classes('cityscapes'),
        palette=get_palette('cityscapes'),
        revise_checkpoint=[(r'^module\.', ''), ('model.', '')])

    # source
    feature_source = []
    with torch.no_grad():
        for i, data_dict in tqdm(enumerate(source_loader)):
            if i > n_emb_source:
                src_cls_score = model.encode_decode(img=data_dict['img'].data[0].cuda(), img_metas=data_dict['img_metas'].data[0])
                src_cls_prob = F.interpolate(torch.softmax(src_cls_score, dim=1), size=(256, 512))
                feature_space = prob_2_entropy(src_cls_prob).squeeze(dim=0).cpu().detach().numpy()
                feature_vector = feature_space.transpose(1, 2, 0).flatten()
                feature_vector_list = feature_vector.tolist()
                feature_source.append(feature_vector_list)

    # target
    feature_target = []
    with torch.no_grad():
        for i, data_dict in tqdm(enumerate(target_loader)):
            if i < n_emb_target:
                tgt_cls_score = model.encode_decode(img=data_dict['img'].data[0].cuda(), img_metas=data_dict['img_metas'].data[0])
                tgt_cls_prob = F.interpolate(torch.softmax(tgt_cls_score, dim=1), size=(256, 512))
                feature_space = prob_2_entropy(tgt_cls_prob).squeeze(dim=0).cpu().detach().numpy()
                feature_vector = feature_space.transpose(1, 2, 0).flatten()
                feature_vector_list = feature_vector.tolist()
                feature_source.append(feature_vector_list)

    feature_all = feature_source + feature_target
    print(len(feature_all))
    feature_all_narry = np.array(feature_all)
    np.save(f"tSNE_feature_{mode}.npy", feature_all_narry)


def main(mode, n_emb_source=300, n_emb_target=300):
    print(f'-------Begin t-SNE for {mode}------')
    data = np.load(f'tSNE_feature_{mode}.npy')
    t0 = time()
    embeddings = TSNE(n_jobs=4).fit_transform(data)
    t1 = time()
    print("t-SNE: %.2g sec" % (t1 - t0))
    np.save(f'embeddings_{mode}', embeddings)
    vis_x_source = embeddings[:, 0][:n_emb_source]
    vis_y_source = embeddings[:, 1][:n_emb_source]

    vis_x_target = embeddings[:, 0][n_emb_source:]
    vis_y_target = embeddings[:, 1][n_emb_source:]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(vis_x_source, vis_y_source, color='#2E75B6', marker='.', label='Source')
    ax.scatter(vis_x_target, vis_y_target, color='#ED7D31', marker='+', label='Target')

    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f't-SNE_{mode}.png', dpi=500)
    print('figure saved')


if __name__ == '__main__':
    data_cfg = '/raid/wzq/code/0-experiment-platform/configs/_base_/datasets/uda_cityscapes_to_acdc_512x512.py'
    model_cfg = '/raid/wzq/code/0-experiment-platform/configs/_base_/models/daformer_sepaspp_mitb5.py'
    checkpoint = '/raid/wzq/code/0-experiment-platform/work_dirs/my_methods/cudaformer/fog/fz_600_12500/model.pth'
    data_preparation(data_cfg, model_cfg, checkpoint, mode='CumFormer_acdc_test', n_emb_source=200, n_emb_target=200)
    main(mode='CumFormer_acdc_test', n_emb_source=200, n_emb_target=200)
