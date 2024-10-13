import torch
import utils
import numpy as np
import random
import os
from torch.utils.tensorboard import SummaryWriter
import time
from torchvision import transforms
import torch_pruning as tp
from model import ResNet

from dataset import read_data, MyDataSet
from NewResNet import resnet18

if __name__ == '__main__':

    #解析撰写的 config.json 文件：
    config = utils.read_config()

    # #设置随机种子
    # utils.set_seed(config['seed'])

    #以当前时间戳的形式保存 tensorboard 日志文件
    t = time.localtime()
    log_path = ('logs/' + str(t.tm_year) + '_' + str(t.tm_mon) + '_' +
                str(t.tm_mday) + '_' + str(t.tm_hour) + '_' + str(t.tm_min) + '_' + str(t.tm_sec))
    os.makedirs(log_path)
    tb_writer = SummaryWriter(log_dir=log_path)

    #读取训练集中所有图像的地址：
    # data_root = "D:\\个人信息\\大三下\\智能系统专业实践\\work6\\ch6_py\\ch6\\ch6_data"
    data_root = config['data_root']
    train_images_path, train_images_label = read_data(root=data_root, data_type='train')
    val_images_path, val_images_label = read_data(root=data_root, data_type='val')

    #读取测试集中所有图像地址的方法与上述相同，请自己撰写。

    #设置批次大小与工作核心数：
    batch_size = config['batch_size']
    nw = config['num_workers']
    # 数据预处理：
    data_transform = {
        "train": transforms.Compose([transforms.Resize([config['img_size'],
                                                    config['img_size']]), transforms.ToTensor()]),
        "val": transforms.Compose([transforms.Resize([config['img_size'],
                                                      config['img_size']]), transforms.ToTensor()])}

    # 构建训练数据集：
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label, transform=data_transform["train"])
    # 构建训练 dataloader：
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                    pin_memory=True, num_workers=nw, drop_last=True)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)

    #测试的数据集构建与 dataloader 构建与之类似，请自己撰写。
    test_dataset = MyDataSet(images_path=val_images_path,
                              images_class=val_images_label, transform=data_transform["val"])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                    pin_memory=True, num_workers=nw, drop_last=True)
    #如果有验证集，也请自己撰写。

    #实例化自己的训练模型：
    num_classes = config['num_classes']
    # 确定数据类型
    device = config['device']
    # 定义模型
    train_model = resnet18(num_classes).to(device)
    # train_model = train_model.cuda() # 将训练模型设置在显卡上

    save_name = config['save_path'] + "model-{}-{}-last.pth".format(0, 0)
    torch.save(train_model.state_dict(), save_name)


    # 加载模型超参数
    # train_model.load_state_dict(torch.load("D:\个人信息\大三下\智能系统专业实践\work6\ch6_py\ch6\save_path\\model-0-last.pth", map_location=device))
    # train_model.eval()
    # print("load model successfully")

    # 模型裁剪
    # 对所有 BN 逐层更新稀疏训练梯度，在 main.py 文件中模型实例化后，遍历模型，找到模型的线性层和最后的输出层，将其在剪枝中忽略。
    ignored_layers = []
    for m in train_model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 7:
            ignored_layers.append(m)
    # 定义一个测试输入：
    example_inputs = torch.randn(1, 3, 224, 224).cuda()
    # 使用上述定义的重要性评估：
    imp = utils.MySlimmingImportance()
    # 依据训练代数，设置迭代次数：
    # 初始化剪枝器：
    pruner = tp.pruner.MetaPruner(
        train_model,
        example_inputs,
        importance=imp,
        iterative_steps=config['prue_epoches'],
        ch_sparsity=0.5,  # 目标稀疏性
        ignored_layers=ignored_layers,
    )

    # 设置优化器：
    if config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(train_model.parameters(), lr=config['learning_rate'],
        betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    elif config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(train_model.parameters(),
        lr=config['learning_rate'],
        momentum=0.9,
        dampening=0,
        weight_decay=0,
        nesterov=False)
    else:
        raise ValueError("Optimizer must be Adam or SGD, got {} ".format(config['optimizer']))

    # 撰写学习率衰减：
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epoches'], eta_min=0)
    # 记录每一代的训练日志，并保存在 tensorboard 文件内：
    tags = ['train_loss', 'train_acc', 'test_loss', 'test_acc', 'learning_rate', 'Params', 'MACs']
    last_test_acc = 0
    # 调用训练脚本进行训练：
    for epoch in range(config['epoches']):

        train_loss, train_acc = utils.trainer(train_model, optimizer, train_loader, device, epoch)
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]['lr'], epoch)
        # 并在每一代训练完成后执行一次学习率衰减：
        scheduler.step()

        if epoch % config['prue_epoches'] == 0:
            if epoch != 0:  # 模型裁剪
                pruner.step()
            macs, nparams = tp.utils.count_ops_and_params(train_model, example_inputs)
            print("Params: {:.2f} M".format(nparams / 1e6))    # 参数量
            print("MACs: {:.2f} G".format(macs / 1e9))             # 计算量
            tb_writer.add_scalar(tags[5], nparams / 1e6, epoch)
            tb_writer.add_scalar(tags[6], macs / 1e9, epoch)

        if epoch % config['test_epoches'] == 0:
            test_loss, test_acc = utils.evaluater(train_model, test_loader, device, epoch)
            tb_writer.add_scalar(tags[2], test_loss, epoch)
            tb_writer.add_scalar(tags[3], test_acc, epoch)
            last_test_acc = test_acc


    # s最后保存最后一次训练得到的模型参数：
    save_name = config['save_path'] + "model-{}-last.pth".format(epoch)
    torch.save(train_model.state_dict(), save_name)