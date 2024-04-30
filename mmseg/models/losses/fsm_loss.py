""" 
-*- coding: utf-8 -*-
    @Time    : 2022/12/29  14:11
    @Author  : AresDrw
    @File    : fsm_loss.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""
from abc import ABC

import torch
import torch.nn as nn


class FogPassFilter_conv1(nn.Module, ABC):
    def __init__(self, inputsize):
        super(FogPassFilter_conv1, self).__init__()
        self.hidden = nn.Linear(inputsize, inputsize // 2)
        self.hidden2 = nn.Linear(inputsize // 2, inputsize // 4)
        self.output = nn.Linear(inputsize // 4, 64)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.hidden(x)
        x = self.leakyrelu(x)
        x = self.hidden2(x)
        x = self.leakyrelu(x)
        x = self.output(x)
        return x


class FogPassFilter_res1(nn.Module, ABC):
    def __init__(self, inputsize):
        super(FogPassFilter_res1, self).__init__()

        self.hidden = nn.Linear(inputsize, inputsize // 8)
        self.output = nn.Linear(inputsize // 8, 64)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.hidden(x)
        x = self.leakyrelu(x)
        x = self.output(x)
        return x


def gram_matrix(tensor):
    d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram


class FSM_Loss(nn.Module, ABC):
    def __init__(self, gpu_id=0):
        super(FSM_Loss, self).__init__()
        self.fogpassfilter1 = FogPassFilter_conv1(inputsize=2080).cuda()
        # fogpassfilter.load_state_dict(torch.load('./pretrained/fogpassfilter1.pth'))
        self.fogpassfilter2 = FogPassFilter_res1(inputsize=32896).cuda()
        # fogpassfilter.load_state_dict(torch.load('./pretrained/fogpassfilter2.pth'))
        for param in self.fogpassfilter1.parameters():
            param.requires_grad = False
        for param in self.fogpassfilter2.parameters():
            param.requires_grad = False

    def forward(self, clear_feat, fog_feat, fsm_weight, batchsize):
        loss_dict = dict(loss=0)
        for idx, layer in enumerate(fsm_weight):
            layer_fsm_loss = 0
            a_feature = clear_feat[layer]
            b_feature = fog_feat[layer]
            na, da, ha, wa = a_feature.size()
            nb, db, hb, wb = b_feature.size()

            if idx == 0:
                fogpassfilter = self.fogpassfilter1
            elif idx == 1:
                fogpassfilter = self.fogpassfilter2

            for batch_idx in range(batchsize):
                a_gram = gram_matrix(a_feature[batch_idx])
                b_gram = gram_matrix(b_feature[batch_idx])

                vector_b_gram = b_gram[torch.triu(
                    torch.ones(b_gram.size()[0], b_gram.size()[1])).requires_grad_() == 1].requires_grad_()
                vector_a_gram = a_gram[torch.triu(
                    torch.ones(a_gram.size()[0], a_gram.size()[1])).requires_grad_() == 1].requires_grad_()

                fog_factor_b = fogpassfilter(vector_b_gram)
                fog_factor_a = fogpassfilter(vector_a_gram)
                half = int(fog_factor_b.shape[0] / 2)

                layer_fsm_loss += fsm_weight[layer] * torch.mean(
                    (fog_factor_b / (hb * wb) - fog_factor_a / (ha * wa)) ** 2) / half / b_feature.size(0)

            loss_dict['loss'] += layer_fsm_loss

        return loss_dict


def calc_fsm_loss(clear_feat, fog_feat, fsm_weight, batchsize):
    loss_fn = FSM_Loss(gpu_id=0)
    return loss_fn(clear_feat, fog_feat, fsm_weight, batchsize)


if __name__ == "__main__":
    clear_features = {'layer0': torch.randn(2, 64, 128, 128).cuda(),  # [B, 64, 128, 128]
                      'layer1': torch.randn(2, 256, 64, 64).cuda()}  # [B, 256, 64, 64]
    fog_features = {'layer0': torch.randn(2, 64, 128, 128).cuda(),
                    'layer1': torch.randn(2, 256, 128, 128).cuda()}  # [B, 256, 64, 64]
    fsm_weights = {'layer0': 0.5, 'layer1': 0.5}
    loss = calc_fsm_loss(clear_features, fog_features, fsm_weights, 2)
    print(loss)
