"""
-*- coding: utf-8 -*-
    @Time    : 2023/2/10  14:19
    @Author  : AresDrw
    @File    : CuDAFormer.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""
import os
from copy import deepcopy
import random
import numpy as np
import torch
from PIL import Image

import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor, build_backbone
from mmseg.models.uda.SAM_enhanced_DAFormer import Merger, make_debug_dirs
from mmseg.models.uda.uda_decorator import UDADecorator
from mmseg.models.utils.dacs_transforms import denorm, get_mean_std, strong_transform, get_class_masks
from mmseg.models.utils.visualization import subplotimg, colorize_mask, Cityscapes_palette

debug_dir = '/raid/wzq/code/0-experiment-platform/work_dirs/SAM_enhanced/cs2adverse/valid_test_debug/new_fog/'


def calc_entropy(prob):
    """
    :param prob: softmax of the score
    :return:
    """
    entropy_map = torch.sum(-prob * torch.log(prob + 1e-7), dim=1)
    return entropy_map


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_model(model):
    for p in model.parameters():
        p.requires_grad = True


@UDA.register_module()
class CuDAFormerEnhancedBySAM(UDADecorator):
    def __init__(self, **cfg):
        super(CuDAFormerEnhancedBySAM, self).__init__(**cfg)
        self.local_iter = 0
        self.true_label = 1
        self.fake_label = 0
        self.max_iters = cfg['max_iters']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.debug_img_interval = cfg['debug_img_interval']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']

        self.teacher_m2t = build_segmentor(deepcopy(cfg['model']))

        self.sam_mask_merger = Merger(mode='iou_only',
                                      sam_cfg=cfg['SAM']['sam_model'],
                                      num_cls=cfg['SAM']['num_cls'],
                                      device=cfg['SAM']['device'],
                                      cls_area_threshold=cfg['SAM']['cls_area_threshold'],
                                      iou_conf_threshold=cfg['SAM']['iou_conf_threshold'],
                                      debug_root=debug_dir)

        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.milestone = cfg['milestone_for_imd_pseudo_label']

    def _init_ema_weights(self):
        for param in self.teacher_m2t.parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.teacher_m2t.parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter, module, module_ema):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(module_ema.parameters(), module.parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = alpha_teacher * ema_param.data + (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg=None,
                      imd_img=None,
                      imd_img_metas=None,
                      gt_semantic_seg_imd=None,
                      target_img=None,
                      target_img_metas=None,
                      return_feat=False):
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device
        means, stds = get_mean_std(img_metas, dev)

        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }

        if self.local_iter == 0:
            self._init_ema_weights()

        if self.local_iter >= 0:
            # use the intermediate domain
            self._update_ema(iter=self.local_iter,
                             module=self.get_model(),
                             module_ema=self.teacher_m2t)

            # 1. Keep training on source images
            clean_losses = self.get_model().forward_train(
                img, img_metas, gt_semantic_seg, return_feat=False)
            clean_losses = add_prefix(clean_losses, 'src')
            clean_loss, clean_log_vars = self._parse_losses(clean_losses)
            log_vars.update(clean_log_vars)
            clean_loss.backward(retain_graph=True)

            with torch.no_grad():
                # generate pseudo labels on the target domain
                target_logits = self.teacher_m2t.encode_decode(target_img, target_img_metas)
                target_softmax = torch.softmax(target_logits.detach(), dim=1)
                pseudo_prob_trg, pseudo_label_trg = torch.max(target_softmax, dim=1)
                ps_large_p_trg = pseudo_prob_trg.ge(self.pseudo_threshold).long() == 1
                ps_size_trg = np.size(np.array(pseudo_label_trg.cpu()))
                pseudo_weight_trg = torch.sum(ps_large_p_trg).item() / ps_size_trg
                pseudo_weight_trg = pseudo_weight_trg * torch.ones(
                    pseudo_prob_trg.shape, device=dev)

                #  add SAM-plugin:
                """
                    (i) change the target_img from [Tensor] -> [np.array]
                    (ii) collect the masks output by sam
                    (iii) write the specific branch for supervised training on enhanced pseudo label
                """
                make_debug_dirs(debug_root=debug_dir, iter=self.local_iter)
                self.sam_mask_merger.set_image_and_get_masks(img=target_img,
                                                             img_metas=target_img_metas,
                                                             debug_iter=self.local_iter)

                tgt_logits = self.get_model().encode_decode(target_img, target_img_metas)
                src_softmax_prob = torch.softmax(tgt_logits, dim=1)
                _, pred_tgt = torch.max(src_softmax_prob, dim=1)

                self.sam_mask_merger.set_pseudo_label_teacher(pl_teacher=pseudo_label_trg,
                                                              pred_tgt_stu=pred_tgt)

                # pseudo_label_new = self.sam_mask_merger.enhance_pseudo_label_by_iou()
                pseudo_label_refined, sam_weight_trg = self.sam_mask_merger.enhance_pseudo_label_debug()
                self.sam_mask_merger.reset()

                # set weight
                if self.psweight_ignore_top > 0:
                    # Don't trust pseudo-labels in regions with potential
                    # rectification artifacts. This can lead to a pseudo-label
                    # drift from sky towards building or traffic light.
                    pseudo_weight_trg[:, :self.psweight_ignore_top, :] = 0
                    sam_weight_trg[:, :self.psweight_ignore_top, :] = 0
                if self.psweight_ignore_bottom > 0:
                    pseudo_weight_trg[:, -self.psweight_ignore_bottom:, :] = 0
                    sam_weight_trg[:, -self.psweight_ignore_bottom:, :] = 0

                # get pseudo_semantic_seg_imd
                imd_softmax = torch.softmax(self.get_model().encode_decode(imd_img, imd_img_metas), dim=1)
                imd_prob, pseudo_semantic_seg_imd = torch.max(imd_softmax, dim=1)

                # Apply mixing
                m2t_mixed_img, m2t_mixed_lbl = [None] * batch_size, [None] * batch_size
                m2t_mix_masks = get_class_masks(pseudo_semantic_seg_imd)
                gt_pixel_weight_trg = torch.ones(pseudo_weight_trg.shape, device=dev)

                for i in range(batch_size):
                    strong_parameters['mix'] = m2t_mix_masks[i]
                    m2t_mixed_img[i], m2t_mixed_lbl[i] = strong_transform(
                        strong_parameters,
                        data=torch.stack((imd_img[i], target_img[i])),
                        target=torch.stack((pseudo_semantic_seg_imd[i], pseudo_label_trg[i])))
                    _, pseudo_weight_trg[i] = strong_transform(
                        strong_parameters,
                        target=torch.stack((gt_pixel_weight_trg[i], pseudo_weight_trg[i])))
                m2t_mixed_img = torch.cat(m2t_mixed_img)
                m2t_mixed_lbl = torch.cat(m2t_mixed_lbl)

            # 2. Train on SAM masks
            # try:
            #     sam_losses = self.get_model().forward_train(
            #         target_img, target_img_metas, pseudo_label_refined, sam_weight_trg, return_feat=False)
            #     sam_losses = add_prefix(sam_losses, 'sam')
            #     sam_loss, sam_log_vars = self._parse_losses(sam_losses)
            #     log_vars.update(sam_log_vars)
            #     sam_loss.backward()
            # except RuntimeError as e:
            #     Image.fromarray((denorm(target_img[0], means, stds) * 255.0).squeeze(dim=0).permute(1, 2, 0).cpu().numpy().astype('uint8')).save(
            #         os.path.join(debug_dir, f'{self.local_iter}_target_img.png'))  # save image
            #     pseudo_label_refined_save = colorize_mask(pseudo_label_refined[0][0].cpu().numpy(),
            #                                               palette=Cityscapes_palette)
            #     pseudo_label_refined_save.save(
            #         os.path.join(debug_dir, f'{self.local_iter}_refined_pl.png'))
            #     plt.imsave(fname=os.path.join(debug_dir, f'{self.local_iter}_weight.png'),
            #                arr=sam_weight_trg[0].cpu().numpy(), cmap='viridis')

            # 3. Train on mixed images
            m2t_mix_losses = self.get_model().forward_train(
                m2t_mixed_img, imd_img_metas, m2t_mixed_lbl, pseudo_weight_trg, return_feat=False)
            m2t_mix_losses = add_prefix(m2t_mix_losses, 'm2t_mix')
            m2t_mix_loss, m2t_mix_log_vars = self._parse_losses(m2t_mix_losses)
            log_vars.update(m2t_mix_log_vars)
            m2t_mix_loss.backward()

        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['base_work_dir'],
                                   'visualization')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_imd_img = torch.clamp(denorm(imd_img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            vis_mixed_img_m2t = torch.clamp(denorm(m2t_mixed_img, means, stds), 0, 1)

            src_logits = self.get_model().encode_decode(img, img_metas)
            src_softmax_prob = torch.softmax(src_logits, dim=1)
            _, pred_src = torch.max(src_softmax_prob, dim=1)

            imd_logits = self.get_model().encode_decode(imd_img, imd_img_metas)
            imd_softmax_prob = torch.softmax(imd_logits, dim=1)
            _, pred_imd = torch.max(imd_softmax_prob, dim=1)

            target_logits = self.get_model().encode_decode(target_img, target_img_metas)
            target_softmax_prob = torch.softmax(target_logits, dim=1)
            entropy_tgt = calc_entropy(target_softmax_prob)
            _, pred_tgt = torch.max(target_softmax_prob, dim=1)

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
                subplotimg(axs[1][0], vis_imd_img[j], 'Intermediate(Ref) Image')
                subplotimg(axs[2][0], vis_trg_img[j], 'Target Image')

                subplotimg(axs[0][1], gt_semantic_seg[j], 'Source Seg GT', cmap='cityscapes')
                subplotimg(axs[1][1], pseudo_semantic_seg_imd[j], 'Imd Pseudo label', cmap='cityscapes')
                subplotimg(axs[2][1], pseudo_label_trg[j], 'Target Pseudo label', cmap='cityscapes')

                subplotimg(axs[0][2], pred_src[j], 'Source Pred', cmap='cityscapes')
                subplotimg(axs[1][2], pred_imd[j], 'Intermediate Imd', cmap='cityscapes')
                subplotimg(axs[2][2], pred_tgt[j], 'Pred Target(Model)', cmap='cityscapes')

                subplotimg(axs[0][3], vis_mixed_img_m2t[j], 'M2t Mixed image')
                subplotimg(axs[1][3], m2t_mix_masks[j][0], 'M2t mixed mask', cmap='gray')
                subplotimg(axs[2][3], m2t_mixed_lbl[j], 'm2t mixed label', cmap='cityscapes')

                subplotimg(axs[0][4], sam_weight_trg[j], 'SAM GT Weight', cmap='viridis')
                subplotimg(axs[1][4], entropy_tgt[j], 'Entropy Target', cmap='viridis')
                subplotimg(axs[2][4], pseudo_label_refined[j], 'SAM Enhanced PL', cmap='cityscapes')

                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                 f'{(self.local_iter + 1):06d}_{j}.png'))
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
