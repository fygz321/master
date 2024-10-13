import json
import os
import random

import numpy as np
import torch
from tqdm import tqdm
import sys
import torch.nn as nn
import torch_pruning as tp
def read_config():
    with open("config.json") as json_file:
        config = json.load(json_file)
    return config['model_config']
#为了保证实验的可复现行，需要设置随机种子，撰写设置随机种子的函数。

def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # 如果用显卡运行，以下两个选项进行设置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # 设置系统环境随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)

# 然后撰写训练脚本 trainer，撰写如下代码：
def trainer(model, optimizer, data_loader, device, epoch):
    '''
    定义训练脚本
    :param model: 训练模型
    :param optimizer: 优化器
    :param data_loader: 数据集
    :param config: 超参数
    :param epoch: 训练当前代数
    :return
    '''
    model.train()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()
    sample_num = 0
    # data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader, 0):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        sample_num += images.shape[0]
        criterion = nn.CrossEntropyLoss() # 定义损失函数
        loss = criterion(output, labels)
        pred_classes = torch.max(output, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels).sum()
        loss.backward()
        accu_loss += loss.detach()

    data_loader.desc = "[train epoch {}] loss: {:.3f}, acc:{:.3f}".format(epoch,
    accu_loss.item() / (step + 1),
    accu_num.item() / (sample_num))

    print("[train epoch {}] loss: {:.3f}, acc:{:.3f}".format(epoch,
    accu_loss.item() / (step + 1),
    accu_num.item() / (sample_num)))

    if not torch.isfinite(loss):
        print('WARNING: non-finite loss, ending training ', loss)
        sys.exit(1)
    optimizer.step()
    optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

#接着撰写测试脚本 evaluater，与训练脚本类似，这里请自己尝试撰写。
def evaluater(model, data_loader, device, epoch):
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        sample_num += images.shape[0]
        criterion = nn.CrossEntropyLoss() # 定义损失函数
        loss = criterion(output, labels)
        pred_classes = torch.max(output, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels).sum()
        accu_loss += loss.detach()

    data_loader.desc = "[test epoch {}] loss: {:.3f}, acc:{:.3f}".format(epoch,
    accu_loss.item() / (step + 1),
    accu_num.item() / (sample_num))

    print("[test epoch {}] loss: {:.3f}, acc:{:.3f}".format(epoch,
    accu_loss.item() / (step + 1),
    accu_num.item() / (sample_num)))

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

# 定义一个评估指标，输出一个一维的重要性得分向量，来评估每个通道的重要性
class MySlimmingImportance(tp.importance.Importance):
    def __call__(self, group, **kwargs):
        # 1. 首先定义一个列表用于存储分组内每一层的重要性
        group_imp = [] # (num_bns, num_channels)
        # 2. 迭代分组内的各个层，对 BN 层计算重要性
        for dep, idxs in group: # idxs 是一个包含所有可剪枝索引的列表， 用于处理 DenseNet 中的局部耦合的情况
            layer = dep.target.module # 获取 nn.Module
            prune_fn = dep.handler # 获取 剪枝函数
            # 3. 对每个 BN 层计算重要性
            if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and layer.affine:
                local_imp = torch.abs(layer.weight.data) # 计算 scale 参数的绝对值大小
                group_imp.append(local_imp) # 将其保存在列表中
        if len(group_imp) == 0: return None # 跳过不包含 BN 层的分组
        # 4. 按通道计算平均重要性
        group_imp = torch.stack(group_imp, dim=0).mean(dim=0)
        return group_imp

# 对 BN 层进行稀疏训练
class MySlimmingPruner(tp.pruner.MetaPruner):
    def regularize(self, model, reg): # 输入参数一般是模型和正则项的权重，这里可以任意修改
        for m in model.modules(): # 遍历所有层
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and m.affine == True:
                m.weight.grad.data.add_(reg * torch.sign(m.weight.data))
