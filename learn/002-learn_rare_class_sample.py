""" 
-*- coding: utf-8 -*-
    @Time    : 2023/2/15  10:40
    @Author  : AresDrw
    @File    : 002-learn_rare_class_sample.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""
import json
import os.path as osp
import torch


def get_rcs_class_probs(sample_class_stats, temperature):
    overall_class_stats = {}
    for s in sample_class_stats:
        s.pop('file')
        for c, n in s.items():
            c = int(c)
            if c not in overall_class_stats:
                overall_class_stats[c] = n
            else:
                overall_class_stats[c] += n
    overall_class_stats = {
        k: v
        for k, v in sorted(
            overall_class_stats.items(), key=lambda item: item[1])
    }
    freq = torch.tensor(list(overall_class_stats.values()))
    freq = freq / torch.sum(freq)
    freq = 1 - freq
    freq = torch.softmax(freq / temperature, dim=-1)

    return list(overall_class_stats.keys()), freq.numpy()


def get_samples_with_classes(sample_with_class, rcs_classes):
    samples_with_class_and_n = {
        int(k): v
        for k, v in sample_with_class.items()
        if int(k) in rcs_classes
    }
    samples_with_class = {}
    for c in rcs_classes:
        samples_with_class[c] = []
        for file, pixels in samples_with_class_and_n[c]:
            if pixels > 3000:
                samples_with_class[c].append(file.split('/')[-1])
        assert len(samples_with_class[c]) > 0
    return samples_with_class


if __name__ == "__main__":
    data_root = '/hy-tmp/datasets/cityscapes'
    with open(osp.join(data_root, 'sample_class_stats.json'), 'r') as of:
        sample_class_stats = json.load(of)

    # 计算稀有类的概率
    rcs_classes, rcs_prob = get_rcs_class_probs(sample_class_stats, temperature=0.01)

    # 拿到类对应的samples
    with open(osp.join(data_root, 'samples_with_class.json'), 'r') as of:
        sample_with_class = json.load(of)

    sample_with_classes = get_samples_with_classes(sample_with_class, rcs_classes)
    print('done')
