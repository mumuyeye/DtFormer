""" 
-*- coding: utf-8 -*-
    @Time    : 2023/8/9  17:49
    @Author  : AresDrw
    @File    : uncertainty_based_prompt.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""
import torch.nn as nn
import torch
from mmseg.models.builder import PROMPTS


@PROMPTS.register_module()
class UncertaintyBasedPrompt(nn.Module):
    """
        Uncertainty based Prompt:

        input:
            cat[x, entropy_map], entropy_map come from Teacher

        architecture:
            conv layers 8 layers

    """

    def __init__(self, in_channels):
        super(UncertaintyBasedPrompt, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=8, out_channels=3, kernel_size=1, stride=1, padding=0),
        )
        self.dropout = nn.Dropout(p=0.4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.sigmoid(self.dropout(x))
        return x


if __name__ == "__main__":
    x = torch.randn(1, 4, 512, 512).cuda()
    model = UncertaintyBasedPrompt(in_channnels=4).cuda()
    out = model(x)
    print('done')
