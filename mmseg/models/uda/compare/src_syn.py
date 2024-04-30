"""
-*- coding: utf-8 -*-
    @Time    : 2023/1/26  13:41
    @Author  : AresDrw
    @File    : fifo.py
    @Software: PyCharm
    @Describe:
-*- encoding:utf-8 -*-
"""
import math
import os
import random
from copy import deepcopy

import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch import optim
from torch.autograd import Variable
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor, build_loss
from mmseg.models.uda.uda_decorator import UDADecorator

from mmseg.models.utils.dacs_transforms import get_mean_std, denorm
from mmseg.models.utils.visualization import subplotimg
from tools.pytorch_metric_learning import losses
from tools.pytorch_metric_learning.distances import CosineSimilarity
from tools.pytorch_metric_learning.reducers import MeanReducer


def calc_entropy(prob):
    """
    :param prob: softmax of the score
    :return:
    """
    entropy_map = torch.sum(-prob * torch.log(prob + 1e-7), dim=1)
    return entropy_map


def gram_matrix(tensor):
    d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram


@UDA.register_module()
class SRCSYN(UDADecorator):
    def __init__(self, **cfg):
        super(SRCSYN, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.num_classes = cfg['model']['decode_head']['num_classes']
        self.ce_loss = build_loss(cfg['model']['decode_head']['loss_decode'])
        self.debug_img_interval = cfg['debug_img_interval']

    def forward_train(self, img, img_metas, gt_semantic_seg=None,
                      imd_img=None, imd_img_metas=None, gt_imd_semantic_seg=None,
                      target_img=None, target_img_metas=None, return_feat=False):
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        means, stds = get_mean_std(img_metas, dev)

        src_losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, return_feat=False)
        src_loss, src_log_vars = self._parse_losses(src_losses)
        log_vars.update(src_log_vars)
        src_loss.backward(retain_graph=False)

        syn_losses = self.get_model().forward_train(
            imd_img, imd_img_metas, gt_imd_semantic_seg, return_feat=False)
        syn_loss, syn_log_vars = self._parse_losses(syn_losses)
        log_vars.update(syn_log_vars)
        syn_loss.backward(retain_graph=False)

        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['base_work_dir'], 'visualization')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_syn_imd_img = torch.clamp(denorm(imd_img, means, stds), 0, 1)
            vis_real_fog_img = torch.clamp(denorm(target_img, means, stds), 0, 1)

            src_logits = self.model.encode_decode(img, img_metas)
            imd_logits = self.model.encode_decode(imd_img, imd_img_metas)
            trg_logits = self.model.encode_decode(target_img, target_img_metas)

            src_softmax_prob = torch.softmax(src_logits, dim=1)
            entropy_src = calc_entropy(src_softmax_prob)
            _, pred_src = torch.max(src_softmax_prob, dim=1)

            imd_softmax_prob = torch.softmax(imd_logits, dim=1)
            entropy_imd = calc_entropy(imd_softmax_prob)
            _, pred_imd = torch.max(imd_softmax_prob, dim=1)

            trg_softmax_prob = torch.softmax(trg_logits, dim=1)
            entropy_trg = calc_entropy(trg_softmax_prob)
            _, pred_trg = torch.max(trg_softmax_prob, dim=1)

            for j in range(batch_size):
                rows, cols = 3, 5
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
                subplotimg(axs[0][0], vis_img[j], 'Source Image')
                subplotimg(axs[1][0], vis_syn_imd_img[j], 'Synthetic foggy Image')
                subplotimg(axs[2][0], vis_real_fog_img[j], 'Real foggy Image')

                subplotimg(axs[0][1], gt_semantic_seg[j], 'Source Seg GT', cmap='cityscapes')
                subplotimg(axs[1][1], gt_imd_semantic_seg[j], 'Synthetic foggy GT', cmap='cityscapes')
                subplotimg(axs[2][1], pred_trg[j], 'Target Seg (Pseudo) GT', cmap='cityscapes')

                subplotimg(axs[0][2], pred_src[j], 'Source Seg Pred', cmap='cityscapes')
                subplotimg(axs[1][2], pred_imd[j], 'Synthetic foggy Pred', cmap='cityscapes')
                subplotimg(axs[2][2], pred_trg[j], 'Foggy Target Pred', cmap='cityscapes')

                subplotimg(axs[0][3], entropy_src[j], 'Source Seg Entropy', cmap='viridis')
                subplotimg(axs[1][3], entropy_imd[j], 'Synthetic foggy Entropy', cmap='viridis')
                subplotimg(axs[2][3], entropy_trg[j], 'Foggy Target Entropy', cmap='viridis')

                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(os.path.join(out_dir, f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()

        self.local_iter += 1
        return log_vars

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs
