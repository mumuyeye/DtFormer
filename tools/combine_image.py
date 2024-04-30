""" 
-*- coding: utf-8 -*-
    @Time    : 2023/10/22  16:28
    @Author  : AresDrw
    @File    : combine_image.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import imageio
import tifffile as tiff
from scipy import misc

rgb_image_path = '/hy-tmp/datasets/TIF/rgb2_8bit.tif'
rgb_16_bit_path = '/hy-tmp/datasets/TIF/rgb2_16bit.tif'
n_in_sh_in_16bit_path = '/hy-tmp/datasets/TIF/near_infr_short_infr_16bit.tif'


def read_image(path):
    image = Image.open(path)
    image_tensor = transforms.ToTensor()(image)
    print('done')


from PIL import Image
import numpy as np

def split_image_tif(image_path, h, w):
    # 打开TIF图像
    image = Image.open(image_path)

    # 将图像转换为NumPy数组
    image_array = np.array(image)

    # 获取图像的高度和宽度
    height, width = image_array.shape[:2]

    # 计算切分后图像的行数和列数
    rows = height // h
    cols = width // w

    # 遍历每一行和每一列，切分图像并保存为PNG文件
    for r in range(rows):
        for c in range(cols):
            # 计算当前切分图像的起始和结束行列索引
            start_row = r * h
            end_row = start_row + h
            start_col = c * w
            end_col = start_col + w

            # 提取当前切分图像的数据
            sub_image_array = image_array[start_row:end_row, start_col:end_col]

            # 创建PIL图像对象
            sub_image = Image.fromarray(sub_image_array)

            # 将切分图像保存为PNG文件
            sub_image.save(f"sub_image_{r}_{c}.png")


# 示例用法
image_path = "input.tif"  # 输入TIF图像的路径
h = 100  # 切分后图像的高度
w = 100  # 切分后图像的宽度

split_image_tif(image_path, h, w)


def combine_band(path1, path2, path3):
    image1 = Image.open(path1)
    image2 = Image.open(path2)
    image3 = Image.open(path3)

    # 转换为Tensor
    transform = transforms.ToTensor()
    image1_tensor = transform(image1)
    image2_tensor = transform(image2)
    image3_tensor = transform(image3)

    # 在第一维度上进行拼接
    concatenated_tensor = torch.cat([image1_tensor, image2_tensor, image3_tensor], dim=0)

    # 创建一个空的3D数组来保存tensor的每个通道作为tif文件的单个帧
    tif_stack = np.zeros((concatenated_tensor.shape[1], concatenated_tensor.shape[2], concatenated_tensor.shape[0]),
                         dtype=np.float32)

    # 将tensor的每个通道复制到tif_stack中的单个帧
    for i in range(concatenated_tensor.shape[2]):
        try:
            tif_stack[:, :, i] = concatenated_tensor[i]
        except IndexError as e:
            pass

    # 以.tif格式保存tif_stack
    imageio.imsave('tensor.tif', tif_stack)
    # tiff.imsa('tensor.tif', tif_stack)


combine_band(path1=rgb_image_path, path2=rgb_16_bit_path, path3=n_in_sh_in_16bit_path)

read_image('tensor.tif')
