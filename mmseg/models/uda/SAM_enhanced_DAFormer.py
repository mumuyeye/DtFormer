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

import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd
from typing import List, Dict

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor, build_backbone
from mmseg.models.losses.preceptual_loss import VGGLoss, VGGLoss_for_trans
from mmseg.models.uda.uda_decorator import UDADecorator
from mmseg.models.utils.dacs_transforms import denorm, get_mean_std, strong_transform, get_class_masks
from mmseg.models.utils.visualization import subplotimg, colorize_mask, Cityscapes_palette
from tools.segment_anything import sam_model_registry

from tools.segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator

from tools.segment_anything.scripts.amg import write_masks_to_folder

debug_dir = '/raid/wzq/code/0-experiment-platform/work_dirs/SAM_enhanced/cs2adverse/valid_test_debug/new_fog/'


def make_debug_dirs(debug_root, iter):
    debug_iter = os.path.join(debug_root, f'iter_{iter}')
    original_cls_mask = os.path.join(debug_iter, 'original_cls_mask')

    # for iou_based
    intersect = os.path.join(debug_root, f'iter_{iter}', 'intersect')
    sam_original_mask = os.path.join(debug_root, f'iter_{iter}', 'sam_original_mask')
    sam_iou_mask = os.path.join(debug_iter, 'sam_iou_mask')
    sam_cls_mask = os.path.join(debug_root, f'iter_{iter}', 'sam_cls_mask')

    # for voting based
    voting_mask = os.path.join(debug_iter, 'voting_cls_mask')

    os.makedirs(debug_iter, exist_ok=True)
    os.makedirs(original_cls_mask, exist_ok=True)
    os.makedirs(intersect, exist_ok=True)
    os.makedirs(sam_original_mask, exist_ok=True)
    os.makedirs(sam_iou_mask, exist_ok=True)
    os.makedirs(sam_cls_mask, exist_ok=True)
    os.makedirs(voting_mask, exist_ok=True)


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


def replace_values(dst, src, ignore_index=-1):
    mask = (dst == ignore_index)
    src = src.long()
    dst[mask] = src[mask]
    return dst


def fill_nonzero_values(src_tensor, tgt_tensor):
    """
    填充张量tgt_tensor中与src_tensor相同位置的值。
    使用src_tensor中不为0的值填充tgt_tensor中相同位置的值。
    :param src_tensor: 源张量: [1, 1, 1024, 2048]
    :param tgt_tensor: 目标张量: [1, 1024, 2048]
    :return: 填充后的目标张量tgt_tensor
    """
    mask = torch.nonzero(src_tensor)
    values = src_tensor[mask[:, 0], mask[:, 1], mask[:, 2]].long()
    tgt_tensor[mask[:, 0], mask[:, 1], mask[:, 2]] = values
    return tgt_tensor


def preprocess(img: torch.Tensor,
               img_metas: str) -> np.ndarray:
    """
    :param img: [B, 3, 512, 512]
    :param img_metas:
    :return:
    """
    means, stds = get_mean_std(img_metas=img_metas, dev=img.device)
    nd_img = (denorm(img[0], means, stds) * 255.0).squeeze(dim=0).permute(1, 2, 0).cpu().numpy().astype('uint8')
    return nd_img


def process_sam_masks(sam_masks: List[Dict], device: str) -> List[Dict]:
    """
    :param sam_masks:
        List[
            dict:{
                    'segmentation': bool ndarray[H, W]
                    'area': int
                    'bbox': List[x_min, y_min, x_max, y_max]
                    'predicted_iou': float
                    'point_coords': List[X,Y] ????
                    'stability_score': float
                    'crop_box': List[0, 0, H, W]
                }
            ]
    :return:
    """
    sam_segments = []
    for sam_mask in sam_masks:
        sam_segments.append(torch.from_numpy(sam_mask['segmentation']).long().unsqueeze(dim=0).to(device))
    return sam_segments


def judge_only(param, semantic_info):
    pass


class Merger(object):
    def __init__(self,
                 mode,
                 sam_cfg,
                 device,
                 num_cls=19,
                 cls_area_threshold=1e4,
                 iou_conf_threshold=0.8,
                 debug_root=None):
        """
        :param mode:
            (i)   merge by iou
            (ii)  merge by voting
            (iii) merge by voting weight
        :return:
        """
        self.mode = mode
        self.sam = SamAutomaticMaskGenerator(
            model=sam_model_registry[sam_cfg['model_type']](checkpoint=sam_cfg['checkpoint']).to(device),
            points_per_side=sam_cfg['points_per_side'],
            pred_iou_thresh=sam_cfg['pred_iou_thresh'],
            stability_score_thresh=sam_cfg['stability_score_thresh'],
            crop_n_layers=sam_cfg['crop_n_layers'],
            crop_n_points_downscale_factor=sam_cfg['crop_n_points_downscale_factor'],
            min_mask_region_area=sam_cfg['min_mask_region_area'])
        self.num_cls = num_cls
        self.cls_area_threshold = cls_area_threshold
        self.iou_conf_threshold = iou_conf_threshold
        self.debug_root = debug_root

    def set_image_and_get_masks(self, img: torch.Tensor, img_metas: str, debug_iter: int):
        self.debug_iter = debug_iter
        self.img = preprocess(img, img_metas)
        self.img_h = img.size()[2]
        self.img_w = img.size()[3]
        self.dev = img.device

        # debug
        sam_masks = self.sam.generate(self.img)

        sorted_sam_masks = sorted(sam_masks, key=(lambda x: x['area']), reverse=True)

        height, width = sam_masks[0]['segmentation'].shape
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for i, mask in enumerate(sorted_sam_masks):
            random_color = tuple(np.random.randint(0, 256, 3))
            bool_mask = mask['segmentation']
            mask_rgb = np.zeros((height, width, 3), dtype=np.uint8)
            mask_rgb[bool_mask] = random_color
            image += mask_rgb

        Image.fromarray(image).save(
            os.path.join(self.debug_root, f'iter_{self.debug_iter}', f'{self.debug_iter}_sam_all.png'))

        write_masks_to_folder(sam_masks, os.path.join(self.debug_root, f'iter_{self.debug_iter}', 'sam_original_mask'))

        self.sam_segments = process_sam_masks(sam_masks=sam_masks,
                                              device=self.dev)
        del sam_masks

    def set_pseudo_label_teacher(self,
                                 pl_teacher: torch.Tensor,
                                 pred_tgt_stu: torch.Tensor):
        """
        :param pl_teacher: [1, img_h, img_w]
        :return:
        """
        self.pl_teacher = pl_teacher
        self.pred_tgt_stu = pred_tgt_stu

    def enhance_pseudo_label_by_iou(self):
        # debug
        debug_iter_dir = os.path.join(self.debug_root, f'iter_{self.debug_iter}')
        Image.fromarray(self.img).save(os.path.join(debug_iter_dir, f'{self.debug_iter}_img.png'))  # save image

        color_stu = colorize_mask(self.pred_tgt_stu[0].cpu().numpy(), palette=Cityscapes_palette)
        color_stu.save(os.path.join(debug_iter_dir, f'{self.debug_iter}_original_student_pred.png'))

        color_pl = colorize_mask(self.pl_teacher[0].cpu().numpy(), palette=Cityscapes_palette)  # save ori_pl_t
        color_pl.save(os.path.join(debug_iter_dir, f'{self.debug_iter}_original_pseudo_label.png'))

        cls_bool_slices = []
        for cls_id in range(self.num_cls):
            cls_mask = (self.pl_teacher == cls_id)
            cls_bool_slices.append(cls_mask)

            # debug
            cls_mask_save = Image.fromarray(cls_mask[0].cpu().numpy().astype('uint8') * 255).convert('P')
            cls_mask_save.save(os.path.join(debug_iter_dir, 'original_cls_mask', f'cls_{cls_id}_mask.png'))

            # initialize the enhanced_pseudo_label
        cls_sam_masks = [[] for _ in range(self.num_cls)]
        enhanced_pl = torch.ones((1, self.img_h, self.img_w), dtype=torch.long).to(self.dev) * -1

        # calculate recall and merge
        for cls_id, cls_bool_slice in enumerate(cls_bool_slices):
            if cls_bool_slice.sum() > self.cls_area_threshold:  # get rid of no-showed classes
                for j, sam_segment in enumerate(self.sam_segments):
                    iou_mask = cls_bool_slice * sam_segment
                    intersect = (cls_bool_slice * sam_segment).sum()
                    recall_pl = intersect / cls_bool_slice.sum()
                    recall_sam = intersect / sam_segment.sum()
                    if recall_sam > self.iou_conf_threshold or recall_pl > self.iou_conf_threshold:
                        cls_sam_masks[cls_id].append(sam_segment)

                        # debug
                        iou_mask_save = Image.fromarray(iou_mask[0].cpu().numpy().astype('uint8') * 255).convert('P')
                        iou_mask_save.save(os.path.join(debug_iter_dir, 'intersect',
                                                        f'cls_{cls_id}_sam_{j}_rsam_{round(float(recall_sam.cpu().numpy()), 3)}_rpl_{round(float(recall_pl.cpu().numpy()), 3)}.png'))
                        sam_segment_save = Image.fromarray(sam_segment[0].cpu().numpy().astype('uint8') * 255).convert(
                            'P')
                        sam_segment_save.save(
                            os.path.join(debug_iter_dir, 'sam_iou_mask',
                                         f'cls_{cls_id}_sam_{j}_rsam_{round(float(recall_sam.cpu().numpy()), 3)}_rpl_{round(float(recall_pl.cpu().numpy()), 3)}.png'))

        # merge the semantic mask in cls_sam_masks
        for cls_id, d in enumerate(cls_sam_masks):
            if len(d) > 0:
                dp = torch.stack(d)
                d_sum = dp.sum(dim=0)
                d_sum[d_sum >= 1] = 1
                cls_sam_masks[cls_id] = d_sum

                # debug
                sam_cls_mask_save = Image.fromarray(
                    cls_sam_masks[cls_id][0].cpu().numpy().astype('uint8') * 255).convert('P')
                sam_cls_mask_save.save(os.path.join(debug_iter_dir, 'sam_cls_mask', f'sam_cls_{cls_id}_mask.png'))

        # merge the cls_sam_masks -> enhanced_pl
        for cls_id, d in enumerate(cls_sam_masks):
            if len(d) > 0:
                if cls_id == 0:
                    enhanced_pl = fill_nonzero_values(src_tensor=d, tgt_tensor=enhanced_pl)
                    enhanced_pl[enhanced_pl == 1] = cls_id
                else:
                    d[d == 1] = cls_id
                    enhanced_pl = fill_nonzero_values(src_tensor=d, tgt_tensor=enhanced_pl)

        # debug
        sam_merge = colorize_mask(enhanced_pl[0].cpu().numpy(), palette=Cityscapes_palette)  # save ori_pl_t
        sam_merge.save(os.path.join(debug_iter_dir, f'{self.debug_iter}_enhanced_by_iou.png'))

        return enhanced_pl  # [1, img_h, img_w]

    def enhance_pseudo_label_teacher_by_voting(self):
        debug_iter_dir = os.path.join(self.debug_root, f'iter_{self.debug_iter}')
        global_valid_mask = torch.zeros_like(self.pl_teacher, device=self.dev).long()

        Image.fromarray(self.img).save(os.path.join(debug_iter_dir, f'{self.debug_iter}_img.png'))  # save image

        color_pl = colorize_mask(self.pl_teacher[0].cpu().numpy(), palette=Cityscapes_palette)  # save ori_pl_t
        color_pl.save(os.path.join(debug_iter_dir, f'{self.debug_iter}_original_pseudo_label.png'))

        global_cls_mask = torch.ones_like(self.pl_teacher, device=self.dev) * 255

        keep_masks = []
        for keep_id in [1, 4, 5, 6, 7]:
            keep_valid_mask = (self.pl_teacher == keep_id)
            keep_masks.append(keep_valid_mask)

        for i, sam_mask in enumerate(self.sam_segments):
            valid_mask = (sam_mask == 1)
            mask_to_vote = self.pl_teacher * sam_mask

            # debug
            mask_to_vote_save = colorize_mask(mask_to_vote[0].cpu().numpy(), palette=Cityscapes_palette)
            mask_to_vote_save.save(
                os.path.join(debug_iter_dir, 'voting_cls_mask', f'{self.debug_iter}_sam_mask_{i}_cls_vote.png'))

            global_valid_mask[valid_mask] = 1

            # postprocessor
            top_1_propose_class_id = torch.bincount(mask_to_vote[valid_mask].flatten()).topk(1).indices
            if top_1_propose_class_id in [0, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]:
                global_cls_mask[valid_mask] = top_1_propose_class_id

        # process the rare classes
        for keep_mask, keep_id in zip(keep_masks, [1, 4, 5, 6, 7]):
            if keep_mask.sum() == 0:
                pass
            else:
                global_cls_mask[keep_mask] = keep_id

        global_cls_mask_save = colorize_mask(global_cls_mask[0].cpu().numpy(), palette=Cityscapes_palette)
        global_cls_mask_save.save(os.path.join(debug_iter_dir, f'{self.debug_iter}_refined_pl_teacher_rc_no_fill.png'))

        global_invalid_mask = (global_valid_mask == 0)
        global_valid_pl_teacher = self.pl_teacher * global_valid_mask
        global_valid_pl_teacher[global_invalid_mask] = -1

        global_valid_pl_teacher_save = colorize_mask(global_valid_pl_teacher[0].cpu().numpy(),
                                                     palette=Cityscapes_palette)
        global_valid_pl_teacher_save.save(os.path.join(debug_iter_dir, f'{self.debug_iter}_select_pl_teacher.png'))

        global_invalid_save = Image.fromarray(global_invalid_mask[0].cpu().numpy() * 255)
        global_invalid_save.save(os.path.join(debug_iter_dir, f'{self.debug_iter}_edge.png'))

        global_cls_mask = replace_values(global_cls_mask, self.pl_teacher, ignore_index=255)

        global_cls_mask_save = colorize_mask(global_cls_mask[0].cpu().numpy(), palette=Cityscapes_palette)
        global_cls_mask_save.save(os.path.join(debug_iter_dir, f'{self.debug_iter}_refined_pl_teacher_rc_fill.png'))

        return global_cls_mask.unsqueeze(dim=0)  # [1, 1, 512, 512]

    def enhance_pseudo_label_by_voting_weight(self):
        # debug original info
        debug_iter_dir = os.path.join(self.debug_root, f'iter_{self.debug_iter}')
        Image.fromarray(self.img).save(os.path.join(debug_iter_dir, f'{self.debug_iter}_img.png'))  # save image
        color_pl = colorize_mask(self.pl_teacher[0].cpu().numpy(), palette=Cityscapes_palette)  # save ori_pl_t
        color_pl.save(os.path.join(debug_iter_dir, f'{self.debug_iter}_original_pseudo_label.png'))

        # 1:ini the masks
        global_cls_mask = torch.ones_like(self.pl_teacher, device=self.dev) * -1
        global_can_be_marked_mask = torch.ones_like(self.pl_teacher, device=self.dev).bool()  # mark the first
        global_weight = torch.zeros_like(self.pl_teacher, device=self.dev).float()

        # 2: run the segments
        for i, sam_mask in enumerate(self.sam_segments):
            valid_mask = (sam_mask == 1)
            mask_to_vote = self.pl_teacher * sam_mask

            # A. calculate current mask which can be marked
            new_mask_to_be_marked = (valid_mask * global_can_be_marked_mask).bool()
            new_mask_to_vote = mask_to_vote * new_mask_to_be_marked
            global_can_be_marked_mask[new_mask_to_be_marked] = False

            # B. count the semantic info in current mask
            semantic_info = torch.bincount(new_mask_to_vote[new_mask_to_be_marked].flatten())
            semantic_info = torch.cat(
                (semantic_info, torch.zeros(19 - len(semantic_info), dtype=torch.long, device=self.dev)))
            num_all_pixels = new_mask_to_be_marked.sum()

            # C. deal with easy classes
            top_1_propose_class_id = semantic_info.topk(1).indices
            top_1_propose_class_prop = semantic_info.topk(1).values / num_all_pixels
            global_cls_mask[new_mask_to_be_marked] = top_1_propose_class_id
            global_weight[new_mask_to_be_marked] = top_1_propose_class_prop

        # debug:
        # no_fill
        global_cls_mask_save = colorize_mask(global_cls_mask[0].cpu().numpy(), palette=Cityscapes_palette)
        global_cls_mask_save.save(
            os.path.join(debug_iter_dir, f'{self.debug_iter}_refined_pl_easy_class.png'))

        # D. deal with road and sidewalk
        # D.1 save road and sidewalk mask
        keep_masks = []
        for keep_id in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            keep_valid_mask = (self.pl_teacher == keep_id)
            keep_masks.append(keep_valid_mask)

        # D.2 stack the road layer
        # get the Road + invalid_mask
        road_sam_mask = (global_cls_mask == 0)
        road_can_be_refined = global_can_be_marked_mask + road_sam_mask
        global_cls_mask[keep_masks[0] * road_can_be_refined] = 0
        global_weight[keep_masks[0] * road_can_be_refined] = 1

        # D.3 stack the sidewalk layer
        sidewalk_can_be_refined = global_can_be_marked_mask + (global_cls_mask == 0)
        global_cls_mask[keep_masks[1] * sidewalk_can_be_refined] = 1
        global_weight[keep_masks[1] * sidewalk_can_be_refined] = 1

        # key classes
        for key_cls in [2, 3, 4, 5, 6, 7, 8, 9]:
            global_cls_mask[keep_masks[key_cls] * global_can_be_marked_mask] = key_cls
            global_weight[keep_masks[key_cls] * global_can_be_marked_mask] = 0.8

        # debug:
        # no_fill
        global_cls_mask_save = colorize_mask(global_cls_mask[0].cpu().numpy(), palette=Cityscapes_palette)
        global_cls_mask_save.save(
            os.path.join(debug_iter_dir, f'{self.debug_iter}_fill_pl_key_class.png'))

        # weight
        plt.imsave(fname=os.path.join(debug_iter_dir, f'{self.debug_iter}_weight.png'),
                   arr=global_weight[0].cpu().numpy(), cmap='viridis')

        # select_pl_teacher
        global_valid_mask = (global_can_be_marked_mask == 0)
        global_valid_pl_teacher = self.pl_teacher * global_valid_mask
        global_valid_pl_teacher[global_can_be_marked_mask == 1] = -1

        global_valid_pl_teacher_save = colorize_mask(global_valid_pl_teacher[0].cpu().numpy(),
                                                     palette=Cityscapes_palette)
        global_valid_pl_teacher_save.save(os.path.join(debug_iter_dir, f'{self.debug_iter}_select_pl_teacher.png'))

        # # fill
        # global_cls_mask = replace_values(global_cls_mask, self.pl_teacher, ignore_index=-1)
        # global_cls_mask_save = colorize_mask(global_cls_mask[0].cpu().numpy(), palette=Cityscapes_palette)
        # global_cls_mask_save.save(
        #     os.path.join(debug_iter_dir, f'{self.debug_iter}_refined_pl_teacher_rc_fill_ori_pl.png'))

        return global_cls_mask.unsqueeze(dim=0), global_weight.squeeze(dim=1)  # [1, 1, 512, 512]

    def enhance_pseudo_label_debug(self):
        enc = self.enhance_pseudo_label_by_iou()
        return self.enhance_pseudo_label_by_voting_weight()

    def reset(self):
        del self.img
        del self.sam_segments
        del self.pl_teacher
        del self.debug_iter


@UDA.register_module()
class DAFormerEnhancedBySAM(UDADecorator):
    def __init__(self, **cfg):
        super(DAFormerEnhancedBySAM, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.debug_img_interval = cfg['debug_img_interval']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.milestone_full_supervised_on_sam = cfg['milestone_full_supervised_on_sam']

        self.ema_model = build_segmentor(deepcopy(cfg['model']))
        self.imnet_model = build_segmentor(deepcopy(cfg['model']))
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

    def _init_ema_weights(self):
        for param in self.ema_model.parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.ema_model.parameters())
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
            # update the parameters will be at the beginning
            self._update_ema(iter=self.local_iter,
                             module=self.get_model(),
                             module_ema=self.ema_model)

            # Train on source images
            clean_losses = self.get_model().forward_train(
                img, img_metas, gt_semantic_seg, return_feat=False)
            clean_loss, clean_log_vars = self._parse_losses(clean_losses)
            log_vars.update(clean_log_vars)
            clean_loss.backward(retain_graph=False)

            # Generate pseudo-label
            with torch.no_grad():
                for m in self.get_model().modules():
                    if isinstance(m, _DropoutNd):
                        m.training = False
                    if isinstance(m, DropPath):
                        m.training = False
                pseudo_logits = self.get_model().encode_decode(
                    target_img, target_img_metas)

                pseudo_softmax = torch.softmax(pseudo_logits.detach(), dim=1)
                pseudo_prob, pseudo_label = torch.max(pseudo_softmax, dim=1)
                ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
                ps_size = np.size(np.array(pseudo_label.cpu()))
                pseudo_weight = torch.sum(ps_large_p).item() / ps_size
                pseudo_weight = pseudo_weight * torch.ones(
                    pseudo_prob.shape, device=dev)

                if self.psweight_ignore_top > 0:
                    # Don't trust pseudo-labels in regions with potential
                    # rectification artifacts. This can lead to a pseudo-label
                    # drift from sky towards building or traffic light.
                    pseudo_weight[:, :self.psweight_ignore_top, :] = 0
                if self.psweight_ignore_bottom > 0:
                    pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
                gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)

                # # TODO: Add SAM's masks here
                # # we also add directly training on the enhanced label
                # """
                #     (i) change the target_img from [Tensor] -> [np.array]
                #     (ii) collect the masks output by sam
                #     (iii) write the specific branch for supervised training on enhanced pseudo label
                # """
                # make_debug_dirs(debug_root=debug_dir, iter=self.local_iter)
                # self.sam_mask_merger.set_image_and_get_masks(img=target_img,
                #                                              img_metas=target_img_metas,
                #                                              debug_iter=self.local_iter)
                # self.sam_mask_merger.set_pseudo_label_teacher(pl_teacher=pseudo_label)
                #
                # # pseudo_label_new = self.sam_mask_merger.enhance_pseudo_label_by_iou()
                # pseudo_label_refined = self.sam_mask_merger.enhance_pseudo_label_teacher_by_voting()
                # self.sam_mask_merger.reset()

                # Apply mixing
                mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
                mix_masks = get_class_masks(gt_semantic_seg)

                for i in range(batch_size):
                    strong_parameters['mix'] = mix_masks[i]
                    mixed_img[i], mixed_lbl[i] = strong_transform(
                        strong_parameters,
                        data=torch.stack((img[i], target_img[i])),
                        target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])))
                    _, pseudo_weight[i] = strong_transform(
                        strong_parameters,
                        target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
                mixed_img = torch.cat(mixed_img)
                mixed_lbl = torch.cat(mixed_lbl)

            # # Train on SAM refined pseudo labels
            # sam_losses = self.get_model().forward_train(
            #     target_img, target_img_metas, pseudo_label_refined, return_feat=False)
            # sam_losses = add_prefix(sam_losses, 'sam')
            # sam_loss, sam_log_vars = self._parse_losses(sam_losses)
            # sam_loss *= 0.1
            # log_vars.update(sam_log_vars)
            # sam_loss.backward(retain_graph=False)

            # Train on mixed images
            mix_losses = self.get_model().forward_train(
                mixed_img, img_metas, mixed_lbl, pseudo_weight, return_feat=True)
            mix_losses.pop('features')
            mix_losses = add_prefix(mix_losses, 'mix')
            mix_loss, mix_log_vars = self._parse_losses(mix_losses)
            log_vars.update(mix_log_vars)
            mix_loss.backward()

        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['base_work_dir'],
                                   'visualization')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)

            src_logits = self.get_model().encode_decode(img, img_metas)
            src_softmax_prob = torch.softmax(src_logits, dim=1)
            entropy_src = calc_entropy(src_softmax_prob)
            _, pred_src = torch.max(src_softmax_prob, dim=1)

            target_logits = self.get_model().encode_decode(target_img, target_img_metas)
            target_softmax_prob = torch.softmax(target_logits, dim=1)
            entropy_tgt = calc_entropy(target_softmax_prob)
            _, pred_tgt = torch.max(target_softmax_prob, dim=1)

            target_pseudo_logits = self.ema_model.encode_decode(target_img, target_img_metas)
            target_pseudo_prob = torch.softmax(target_pseudo_logits, dim=1)
            _, target_pseudo_label = torch.max(target_pseudo_prob, dim=1)

            for j in range(batch_size):
                rows, cols = 2, 6
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
                subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')

                subplotimg(axs[0][1], gt_semantic_seg[j], 'Source Seg GT', cmap='cityscapes')
                subplotimg(axs[1][1], target_pseudo_label[j], 'Target Pseudo label', cmap='cityscapes')

                subplotimg(axs[0][2], pred_src[j], 'Source Pred', cmap='cityscapes')
                subplotimg(axs[1][2], pred_tgt[j], 'Pred Target(Model)', cmap='cityscapes')

                subplotimg(axs[0][3], entropy_src[j], 'Entropy Source', cmap='viridis')
                subplotimg(axs[1][3], entropy_tgt[j], 'Entropy Target', cmap='viridis')

                subplotimg(axs[0][4], vis_mixed_img[j], 'Mixed Image')
                subplotimg(axs[1][4], mixed_lbl[j], 'Mixed label', cmap='cityscapes')

                subplotimg(axs[0][5], pseudo_weight[j], 'Label Weight', vmin=0, vmax=1)
                subplotimg(axs[1][5], pseudo_label_refined[j], 'Enhanced Pseudo label', cmap='cityscapes')

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
