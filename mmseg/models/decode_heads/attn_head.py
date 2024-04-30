""" 
-*- coding: utf-8 -*-
    @Time    : 2024/1/18  19:38
    @Author  : AresDrw
    @File    : attn_head.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""
from copy import deepcopy

import torch
from torch.nn import functional as F

from ...core import add_prefix
from ...ops import resize as _resize
from .. import builder
from ..builder import HEADS
from ..segmentors.hrda_encoder_decoder import crop
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class AttnHead(BaseDecodeHead):

    def __init__(self,
                 original_head,
                 attention_embed_dim=256,
                 attention_classwise=True,
                 fixed_attention=None,
                 debug_output_attention=False,
                 debug=True,
                 **kwargs):
        head_cfg = deepcopy(kwargs)
        attn_cfg = deepcopy(kwargs)
        self.debug_attn = debug
        if original_head == 'DAFormerHead':
            attn_cfg['channels'] = attention_embed_dim
            attn_cfg['decoder_params']['embed_dims'] = attention_embed_dim
            if attn_cfg['decoder_params']['fusion_cfg']['type'] == 'aspp':
                attn_cfg['decoder_params']['fusion_cfg'] = dict(
                    type='conv',
                    kernel_size=1,
                    act_cfg=dict(type='ReLU'),
                    norm_cfg=attn_cfg['decoder_params']['fusion_cfg']
                    ['norm_cfg'])
            kwargs['init_cfg'] = None
            kwargs['input_transform'] = 'multiple_select'
            self.os = 4
        elif original_head == 'DLV2Head':
            kwargs['init_cfg'] = None
            kwargs.pop('dilations')
            kwargs['channels'] = 1
            self.os = 8
        else:
            raise NotImplementedError(original_head)
        super(AttnHead, self).__init__(**kwargs)
        del self.conv_seg
        del self.dropout

        head_cfg['type'] = original_head
        self.head = builder.build_head(head_cfg)

        attn_cfg['type'] = original_head
        if not attention_classwise:
            attn_cfg['num_classes'] = 1
        if fixed_attention is None:
            self.attention = builder.build_head(attn_cfg)
        else:
            self.attention = None
            self.fixed_attention = fixed_attention

        self.debug_output_attention = debug_output_attention

    def get_attention(self, inp):
        if self.attention is not None:
            att = torch.sigmoid(self.attention(inp))
        else:
            att = self.fixed_attention
        return att

    def forward(self, inputs):

        # get original output
        seg = self.head(inputs)

        # get attention output
        att = self.get_attention(inputs)

        # fused seg
        fused_seg = att * seg + seg

        if self.debug_attn:
            if torch.is_tensor(att):
                self.debug_output['Attention'] = torch.sum(
                    att * torch.softmax(fused_seg, dim=1), dim=1,
                    keepdim=True).detach().cpu().numpy()

        return fused_seg, att

    def forward_train(self,
                      inputs,
                      img_metas,
                      gt_semantic_seg,
                      train_cfg,
                      seg_weight=None,
                      return_logits=False):
        """Forward function for training."""
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg, seg_weight)
        if return_logits:
            losses['logits'] = seg_logits
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing, only ``fused_seg`` is used."""
        return self.forward(inputs)[0]

    def losses(self, seg_logit, seg_label, seg_weight=None):
        """Compute losses."""
        fused_seg, att = seg_logit
        loss = super(AttnHead, self).losses(fused_seg, seg_label, seg_weight)

        if self.debug_attn:
            self.debug_output['attn'] = att
            self.debug_output['GT'] = seg_label.squeeze(1).detach().cpu().numpy()
            # Remove debug output from cross entropy loss
            self.debug_output.pop('Seg. Pred.', None)
            self.debug_output.pop('Seg. GT', None)

        return loss
