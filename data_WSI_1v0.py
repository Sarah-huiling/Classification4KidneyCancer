from __future__ import print_function

import os

from torch.utils.data import DataLoader, Dataset, random_split

import torch
import numpy as np
import random
import matplotlib.pyplot as plt

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2
import csv
import xlrd
from torchvision.transforms import transforms, ToPILImage
import scipy.ndimage


def load_label_form_xl(xl_path):
    reads = xlrd.open_workbook(xl_path)
    label = []
    for row in range(reads.sheet_by_index(0).nrows):
        label.append(reads.sheet_by_index(0).cell(row, 0).value)  # 从excel加载PID和labels
        label.append(reads.sheet_by_index(0).cell(row, 1).value)
    return label


def load_npy(x_path):
    label = [0, 0]
    # data = np.load(x_path, allow_pickle=True)
    data = cv2.imread(x_path, cv2.IMREAD_UNCHANGED)

    try:
        if data.shape[1] <= 3:
            data = data.transpose((1, 0, 2))
        if data.shape[2] <= 3:
            data = data.transpose((2, 0, 1))  # 512*512*3 to 3*512*512
        # data = data.transpose((2, 0, 1))  # 512*512*3 to 3*512*512
    except:
        print('x_path:', x_path)
        print('data:', data.shape)

    # labels = load_label_form_xl('/media/zhl/ResearchData/20221025病理/patch/labels.xls')
    #
    # #
    # # id_index2 = x_path.split('/')[-2][:2]
    # id_index = labels.index(x_path)
    # # 标签转化成one-hot形式
    # if labels[id_index + 1] == '1.0':
    #     label = [0, 1]
    # if labels[id_index + 1] == '0.0':
    #     label = [1, 0]
    id_index = x_path.split('-')[-1].split('.')[0]
    if id_index == 'Tumor' or id_index == 'tumor':
        label = [0, 1]
    # if labels[id_index + 1] == 0.0:
    # if id_index == 'Normal':
    if id_index == 'BG' or id_index == 'Normal':
        label = [1, 0]
    # print('='*10)
    # print('data:', data.shape)
    # print('id_index:', id_index)
    # print('label:', label)
    return data, label


class MyDataset(Dataset):
    def __init__(self, root, transform=None):
        self.image_files = np.array(root)  # 加载数据
        self.transform = transform   # 数据增强

    def __getitem__(self, index):  # 返回的是CPU tensor
        x, y = load_npy(self.image_files[index])  # 加载传过来的npy数据
        x = scipy.ndimage.zoom(x, np.array([x.shape[0], 224, 224]) / np.array(x.shape), mode='nearest', order=2)
        # # x = x.astype(np.int16)
        # if x.shape[0] <= 3:
        #     x = x.transpose((1, 2, 0))
        # elif x.shape[1] <= 3:
        #     x = x.transpose((0, 2, 1))
        # # 采用transform做data augmentation之前，图像格式需为PIL image
        # x = ToPILImage()(np.uint8(x)).convert('RGB')  # 转成PIL image之前，x.shape[-1]需要<3，且为uint8
        # # x = ToPILImage()(np.uint8(x))  # 转成PIL image之前，x.shape[-1]需要<3，且为uint8
        # if self.transform is not None:
        #     x = self.transform(x)
        # else:
        #     x = transforms.ToTensor()(x)
        # CPU tensor
        return torch.FloatTensor(x), torch.FloatTensor(y)
        # return x, torch.FloatTensor(y)

    def __len__(self):
        return len(self.image_files)


# cv2.imshow('2',dataset.images[0,:,:,:])
# cv2.waitKey(0)
