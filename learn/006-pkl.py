""" 
-*- coding: utf-8 -*-
    @Time    : 2023/8/10  10:06
    @Author  : AresDrw
    @File    : 006-pkl.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""
import pickle

path = '/hy-tmp/0-experiment-platform/work_dir/tta/debug/res.pkl'

f = open(path, 'rb')

data = pickle.load(f)

print(data)