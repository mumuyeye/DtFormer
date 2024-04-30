""" 
-*- coding: utf-8 -*-
    @Time    : 2022/12/29  22:41
    @Author  : AresDrw
    @File    : entropy.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""
import torch
import matplotlib.pyplot as plt


def calc_entropy(prob):
    """
    :param prob: softmax of the score
    :return:
    """
    entropy_map = torch.sum(-prob * torch.log(prob + 1e-7), dim=1)
    return entropy_map


if __name__ == "__main__":
    prob = torch.softmax(torch.randn(1, 19, 256, 512), dim=1)
    ent = calc_entropy(prob).squeeze(dim=0)  # [1, 128, 128]
    plt.imsave('ent.png', ent.numpy(), cmap='viridis')
    print('done')
