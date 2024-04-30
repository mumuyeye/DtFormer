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

from mmseg.models.uda.compare.aux_modules.fogpassfilter import FogPassFilter
from mmseg.models.uda.xformer_YuXin import calc_entropy

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
class FIFO(UDADecorator):
    def __init__(self, **cfg):
        super(FIFO, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.num_classes = cfg['model']['decode_head']['num_classes']
        self.ce_loss = build_loss(cfg['model']['decode_head']['loss_decode'])
        self.debug_img_interval = cfg['debug_img_interval']
        self.lambda_fsm = 0.0000001
        self.lambda_con = 0.0001
        self.fpf_loss = losses.ContrastiveLoss(
            pos_margin=0.1,
            neg_margin=0.1,
            distance=CosineSimilarity(),
            reducer=MeanReducer()
        )

        # set fog-pass-filter
        self.fog_pass_filter = FogPassFilter(input_size_0=cfg['fog_pass_filter']['in_size_0'],
                                             input_size_1=cfg['fog_pass_filter']['in_size_1'])
        self.fog_pass_filter.train()
        self.fog_pass_filter.cuda()
        self.optimizer_fpf_layer0 = optim.Adam(self.fog_pass_filter.filter0.parameters(),
                                               lr=cfg['lr_fpf_params']['lr'],
                                               betas=(0.9, 0.99))
        self.optimizer_fpf_layer1 = optim.Adam(self.fog_pass_filter.filter1.parameters(),
                                               lr=cfg['lr_fpf_params']['lr'],
                                               betas=(0.9, 0.99))
        self.optimizer_fpf_layer0.zero_grad()
        self.optimizer_fpf_layer1.zero_grad()

    def forward_train(self, img, img_metas, gt_semantic_seg=None,
                      imd_img=None, imd_img_metas=None, gt_imd_semantic_seg=None,
                      target_img=None, target_img_metas=None, return_feat=False):
        """
        :param img: source domain
        :param img_metas:
        :param gt_semantic_seg:
        :param imd_img: synthetic fog domain, i.e., Foggy Cityscapes
        :param imd_img_metas:
        :param gt_imd_semantic_seg:
        :param target_img: ACDC or foggy zurich
        :param target_img_metas:
        :param return_feat:

        Here, the
            ``img'' from source domain is correspond to ``cw(clear weather)'' images
            ``imd_img'' from synthetic domain is correspond to ``sf(synthetic fog)'' images
            ``target_img'' from foggy domain is correspond to ``rf(real fog)'' images
        :return:

        The resnet features are:
            [
                [B, 256, 128, 128] # 64
                [B, 512, 128, 128]  # 128
                [B, 1024, 128, 128]
                [B, 2048, 128, 128]
            ]
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        means, stds = get_mean_std(img_metas, dev)

        # 2. train segmentation network
        # don't accumulate grads in fog_pass_filter
        for param in self.fog_pass_filter.parameters():
            param.requires_grad = False

        fsm_weights = {'layer0': 0.5, 'layer1': 0.5}
        if self.local_iter % 3 == 0:
            """
                TODO:
                1. originally use the 5th layer feature
                    overwrite the ``encode_decode'' method in RefineNet
                2. crop the same size, which should be write the ``PairCityscapesDataset''
                    and apply overwriten PairRandomCrop
            """
            syn_imd_logits = self.model.encode_decode(imd_img, imd_img_metas, return_feat=True)
            syn_imd_features = syn_imd_logits.pop('features')
            # syn_imd_log_softmax = torch.log_softmax(syn_imd_logits['out'], dim=1)  # why

            src_logits = self.model.encode_decode(img, imd_img_metas, return_feat=True)
            src_features = src_logits.pop('features')
            src_softmax = torch.softmax(src_logits['out'], dim=1)

            # con_losses['loss'] += torch.nn.KLDivLoss(reduction='batchmean')(syn_imd_log_softmax,
            #                                                                 src_softmax) * self.lambda_con
            # con_losses = add_prefix(con_losses, 'con')
            # con_loss, con_log_vars = self._parse_losses(con_losses)
            # log_vars.update(con_log_vars)
            # con_loss.backward(retain_graph=True)
            src_losses = self.model.forward_train(
                img, img_metas, gt_semantic_seg, return_feat=False)
            src_losses = add_prefix(src_losses, 'src')
            src_loss, src_log_vars = self._parse_losses(src_losses)
            log_vars.update(src_log_vars)
            src_loss.backward(retain_graph=False)

            fsm_weights = {'layer0': 0.5, 'layer1': 0.5}
            sf_features = {'layer0': syn_imd_features[0], 'layer1': syn_imd_features[1]}
            cw_features = {'layer0': src_features[0], 'layer1': src_features[1]}

        elif self.local_iter % 3 == 1:
            #  syn fog and real fog
            syn_imd_logits = self.model.encode_decode(imd_img, imd_img_metas, return_feat=True)
            syn_imd_features = syn_imd_logits.pop('features')

            target_imd_logits = self.model.encode_decode(target_img, target_img_metas, return_feat=True)
            target_imd_features = target_imd_logits.pop('features')

            syn_losses = self.model.forward_train(
                imd_img, imd_img_metas, gt_imd_semantic_seg, return_feat=False)
            syn_losses = add_prefix(syn_losses, 'syn')
            syn_loss, syn_log_vars = self._parse_losses(syn_losses)
            log_vars.update(syn_log_vars)
            syn_loss.backward(retain_graph=False)

            rf_features = {'layer0': target_imd_features[0], 'layer1': target_imd_features[1]}
            sf_features = {'layer0': syn_imd_features[0], 'layer1': syn_imd_features[1]}

        elif self.local_iter % 3 == 2:
            # clear weather and real fog
            src_logits = self.model.encode_decode(img, imd_img_metas, return_feat=True)
            src_features = src_logits.pop('features')

            target_img_logits = self.model.encode_decode(target_img, target_img_metas, return_feat=True)
            target_img_features = target_img_logits.pop('features')

            src_losses = self.model.forward_train(
                img, img_metas, gt_semantic_seg, return_feat=False)
            src_losses = add_prefix(src_losses, 'src')
            src_loss, src_log_vars = self._parse_losses(src_losses)
            log_vars.update(src_log_vars)
            src_loss.backward(retain_graph=False)

            rf_features = {'layer0': target_img_features[0], 'layer1': target_img_features[1]}
            cw_features = {'layer0': src_features[0], 'layer1': src_features[1]}

        fsm_losses = dict(loss=0)
        for idx, layer in enumerate(fsm_weights):
            # fog pass filter loss between different fog conditions a and b
            if self.local_iter % 3 == 0:
                a_feature = cw_features[layer]
                b_feature = sf_features[layer]
            if self.local_iter % 3 == 1:
                a_feature = rf_features[layer]
                b_feature = sf_features[layer]
            if self.local_iter % 3 == 2:
                a_feature = rf_features[layer]
                b_feature = cw_features[layer]

            na, da, ha, wa = a_feature.size()
            nb, db, hb, wb = b_feature.size()

            if idx == 0:
                fogpassfilter = self.fog_pass_filter.filter0
            elif idx == 1:
                fogpassfilter = self.fog_pass_filter.filter1

            fogpassfilter.eval()

            layer_fsm_loss = 0
            for batch_idx in range(batch_size):
                b_gram = gram_matrix(b_feature[batch_idx])
                a_gram = gram_matrix(a_feature[batch_idx])

                if self.local_iter % 3 == 1 or self.local_iter % 3 == 2:
                    a_gram = a_gram * (hb * wb) / (ha * wa)

                vector_b_gram = b_gram[torch.triu(
                    torch.ones(b_gram.size()[0], b_gram.size()[1])).requires_grad_() == 1].requires_grad_()
                vector_a_gram = a_gram[torch.triu(
                    torch.ones(a_gram.size()[0], a_gram.size()[1])).requires_grad_() == 1].requires_grad_()

                fog_factor_b = fogpassfilter(vector_b_gram)
                fog_factor_a = fogpassfilter(vector_a_gram)
                half = int(fog_factor_b.shape[0] / 2)

                layer_fsm_loss += fsm_weights[layer] * torch.mean(
                    (fog_factor_b / (hb * wb) - fog_factor_a / (ha * wa)) ** 2) / half / b_feature.size(0)

            fsm_losses['loss'] += (layer_fsm_loss / 4) * self.lambda_fsm
        fsm_losses = add_prefix(fsm_losses, 'fsm')
        fsm_loss, fsm_log_vars = self._parse_losses(fsm_losses)
        log_vars.update(fsm_log_vars)
        fsm_loss.backward(retain_graph=False)

        # 2 train the fog-pass-filter
        # don't accumulate grads in segmentor
        for param in self.fog_pass_filter.parameters():
            param.requires_grad = True

        feature_cw0, feature_cw1, feature_cw2, feature_cw3, feature_cw4 = self.model.extract_feat(img)
        feature_sf0, feature_sf1, feature_sf2, feature_sf3, feature_sf4 = self.model.extract_feat(imd_img)
        feature_rf0, feature_rf1, feature_rf2, feature_rf3, feature_rf4 = self.model.extract_feat(target_img)

        fsm_weights = {'layer0': 0.5, 'layer1': 0.5}
        sf_features = {'layer0': feature_sf0, 'layer1': feature_sf1}
        cw_features = {'layer0': feature_cw0, 'layer1': feature_cw1}
        rf_features = {'layer0': feature_rf0, 'layer1': feature_rf1}

        fpf_losses = dict(loss=0)
        for idx, layer in enumerate(fsm_weights):
            cw_feature = cw_features[layer]
            sf_feature = sf_features[layer]
            rf_feature = rf_features[layer]

            if idx == 0:
                fogpassfilter = self.fog_pass_filter.filter0
            elif idx == 1:
                fogpassfilter = self.fog_pass_filter.filter1

            self.fog_pass_filter.train()

            sf_gram = [0] * batch_size
            cw_gram = [0] * batch_size
            rf_gram = [0] * batch_size
            vector_sf_gram = [0] * batch_size
            vector_cw_gram = [0] * batch_size
            vector_rf_gram = [0] * batch_size
            fog_factor_sf = [0] * batch_size
            fog_factor_cw = [0] * batch_size
            fog_factor_rf = [0] * batch_size

            for batch_idx in range(batch_size):
                sf_gram[batch_idx] = gram_matrix(sf_feature[batch_idx])
                cw_gram[batch_idx] = gram_matrix(cw_feature[batch_idx])
                rf_gram[batch_idx] = gram_matrix(rf_feature[batch_idx])

                vector_sf_gram[batch_idx] = Variable(sf_gram[batch_idx][torch.triu(
                    torch.ones(sf_gram[batch_idx].size()[0], sf_gram[batch_idx].size()[1])) == 1],
                                                     requires_grad=True)
                vector_cw_gram[batch_idx] = Variable(cw_gram[batch_idx][torch.triu(
                    torch.ones(cw_gram[batch_idx].size()[0], cw_gram[batch_idx].size()[1])) == 1],
                                                     requires_grad=True)
                vector_rf_gram[batch_idx] = Variable(rf_gram[batch_idx][torch.triu(
                    torch.ones(rf_gram[batch_idx].size()[0], rf_gram[batch_idx].size()[1])) == 1],
                                                     requires_grad=True)

                fog_factor_sf[batch_idx] = fogpassfilter(vector_sf_gram[batch_idx])
                fog_factor_cw[batch_idx] = fogpassfilter(vector_cw_gram[batch_idx])
                fog_factor_rf[batch_idx] = fogpassfilter(vector_rf_gram[batch_idx])

            fog_factor_embeddings = torch.cat((torch.unsqueeze(fog_factor_sf[0], 0),
                                               torch.unsqueeze(fog_factor_cw[0], 0),
                                               torch.unsqueeze(fog_factor_rf[0], 0)), 0)

        fog_factor_embeddings_norm = torch.norm(fog_factor_embeddings, p=2, dim=1).detach()
        size_fog_factor = fog_factor_embeddings.size()
        fog_factor_embeddings = fog_factor_embeddings.div(
            fog_factor_embeddings_norm.expand(size_fog_factor[1], batch_size * 3).t())
        fog_factor_labels = torch.LongTensor([0, 1, 2])
        fog_pass_filter_loss = self.fpf_loss(fog_factor_embeddings, fog_factor_labels)

        fpf_losses['loss'] += fog_pass_filter_loss
        fpf_losses = add_prefix(fpf_losses, 'fpf')
        fpf_loss, fpf_log_vars = self._parse_losses(fpf_losses)
        log_vars.update(fpf_log_vars)
        fpf_loss.backward(retain_graph=False)

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
        self.optimizer_fpf_layer0.zero_grad()
        self.optimizer_fpf_layer1.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()
        self.optimizer_fpf_layer0.step()
        self.optimizer_fpf_layer1.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs
