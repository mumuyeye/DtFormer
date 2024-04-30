""" 
-*- coding: utf-8 -*-
    @Time    : 2023/1/27  17:31
    @Author  : AresDrw
    @File    : NoUDA.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""
import os

import torch
from matplotlib import pyplot as plt
from mmcv.runner import load_checkpoint

from mmseg.core import add_prefix
from mmseg.models.builder import UDA
from mmseg.models.uda.uda_decorator import UDADecorator
from mmseg.models.utils.dacs_transforms import get_mean_std, denorm
from mmseg.models.utils.visualization import subplotimg


def calc_entropy(prob):
    """
    :param prob: softmax of the score
    :return:
    """
    entropy_map = torch.sum(-prob * torch.log(prob + 1e-7), dim=1)
    return entropy_map


def freeze_parameters(model):
    for name, p in model.named_parameters():
        p.requires_grad = False
        if 'cls_adapter' in name:
            p.requires_grad = True


@UDA.register_module()
class NoUDA(UDADecorator):
    def __init__(self, **cfg):
        super(NoUDA, self).__init__(**cfg)
        """
            only trained on source domain.
        """
        self.local_iter = 0
        self.debug_img_interval = cfg['debug_img_interval']
        # load_checkpoint(self.get_model(),
        #                 filename=cfg['lsm_config']['checkpoint'],
        #                 strict=False)
        # if cfg['lsm_config']['finetune']:
        #     freeze_parameters(self.get_model())

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg=None,
                      target_img=None,
                      target_img_metas=None,
                      return_feat=False):
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device
        means, stds = get_mean_std(img_metas, dev)

        # Train on source images, no matter the iter changes
        clean_losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, return_feat=False)
        clean_losses = add_prefix(clean_losses, 'src')
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        clean_loss.backward(retain_graph=False)

        seg_debug = self.get_model().decode_head.debug_output
        atten_map = seg_debug['attn']
        atten_cls_map = seg_debug['Attention']

        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['base_work_dir'], 'visualization')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            src_logits = self.get_model().encode_decode(img, img_metas)
            src_softmax_prob = torch.softmax(src_logits, dim=1)
            entropy = calc_entropy(src_softmax_prob)
            attn = torch.mean(atten_map, dim=1)
            _, pred_src = torch.max(src_softmax_prob, dim=1)

            for j in range(batch_size):
                rows, cols = 2, 3
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
                subplotimg(axs[0][1], gt_semantic_seg[j], 'Source Seg GT', cmap='cityscapes')
                subplotimg(axs[0][2], entropy[j], 'Entropy', cmap='viridis')
                subplotimg(axs[1][0], pred_src[j], 'Source Pred', cmap='cityscapes')
                subplotimg(axs[1][1], attn[j], 'Original atten', cmap='gray', vmin=0, vmax=1)
                subplotimg(axs[1][2], atten_cls_map[j], 'Cls atten', cmap='gray', vmin=0, vmax=1)

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
