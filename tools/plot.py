""" 
-*- coding: utf-8 -*-
    @Time    : 2023/2/4  15:19
    @Author  : AresDrw
    @File    : plot.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.patches as mpatches

mpl.rcParams["font.family"] = 'Arial'
mpl.rcParams["mathtext.fontset"] = 'cm'

if __name__ == "__main__":
    fig = plt.figure(figsize=(6, 4))
    ax = fig.gca()

    # x = [i / 10 for i in range(1, 11)]
    # y = [71.3, 71.6, 72.1, 72.7, 73.2,
    #      72.8, 72.2, 71.7, 71.1, 70.9]
    #
    # ax.set_ylim(70.5, 73.6)

    x = [i / 100 for i in [50, 55, 60, 65, 70,
                           75, 80, 85, 90, 95]]
    y = [70.9, 71.1, 71.2, 71.7, 72.0,
         72.4, 72.8, 73.2, 72.9, 72.5]

    ax.set_ylim(70.5, 73.6)

    ax.bar(x, y, width=0.035, color='#EE9572')
    ax.set_xlabel('Stability score threshold $\delta_{iou}$ (fix $\delta_{sta}$=0.8)')
    ax.set_ylabel('mIoU %')
    plt.xticks(x)
    # plt.show()
    # ax.legend()
    plt.tight_layout()
    plt.savefig(r'C:\Users\17138\Desktop\05-论文撰写\08-SAM-EDA\02-图片文件夹\05-sam-stable\01-iou.pdf')
