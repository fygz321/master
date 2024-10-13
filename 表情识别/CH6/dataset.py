from PIL import Image
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os
import json
import random


# 数据读取、参考实验三构建训练集测试集
def read_data(root = './ch6_data', data_type = 'train'):
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    root = root + "\\" + data_type

    # 遍历文件夹，一个文件夹对应一个类别，获取文件夹名，all_class 为[“cow”, “sheep”]
    all_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root,cla))]
    all_class.sort() # 排序，保证顺序一致，保证文件夹顺序一致
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(all_class))  # v 是序号，k 是类别名
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)  # json 格式：“0”：“cow”，“1”:“sheep”
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)  # 存储到硬盘

    # 以下整理图像的路径
    images_path = []  # 存储所有图像路径 ， 如‘sample\\cow\\cow.0.jpeg’, ……
    images_label = []  # 存储对应标签信息，如：0,0,0……，与图像路径文件一一对应
    supported = [".jpg", ".JPG", ".png", ".PNG", ".jpeg"]  # 支持的文件后缀类型

    # 遍历每个文件夹下的文件
    for cla in all_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取 supported 支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path) if os.path.splitext(i)[-1] in supported]
        # 例 如 ：sample\\cow\\cow.0.jpeg,…，images放cow或sheep路径下的所有图像具体路径
        # 获取该类别对应的标签
        image_class = class_indices[cla]

        for img_path in images:
            images_path.append(img_path)
            images_label.append(image_class)

    print("images for " + data_type + ":" + str(len(images_path)))
    return images_path, images_label


class MyDataSet(Dataset):
    '''自定义数据集'''
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)  # 图像数目

    def __getitem__(self, item):
        img = Image.open(self.images_path[item]).convert('RGB')  # 确保三通道，根据模型输入要求
        # RGB 为彩色图像，L 为灰度图像
        label = self.images_class[item]
        # 图像预处理
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)  # 图像分批为 Batch
        labels = torch.as_tensor(labels)
        return images, labels