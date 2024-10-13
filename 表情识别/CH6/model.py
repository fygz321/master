import torch
from torch import nn
from torch.nn import functional as F

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_lxlconv=False, strides=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1) # 高宽不变

        if use_lxlconv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)

        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

class ResNet(nn.Module):
    def __init__(self, Residual, num_classes=7):
        super(ResNet, self).__init__()
        # ResNet前两层相较于GoogLeNet的不同:每个卷积层后增加BN层
        self.b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), # 7,2,3  参数1为通道数，灰度图为1，彩色图为3
                           nn.BatchNorm2d(64),
                           nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.b2 = nn.Sequential(*self.resnet_block(Residual, 64, 64, 3, first_block=True))
        self.b3 = nn.Sequential(*self.resnet_block(Residual, 64, 128, 4))
        self.b4 = nn.Sequential(*self.resnet_block(Residual, 128, 256, 6))
        self.b5 = nn.Sequential(*self.resnet_block(Residual, 256, 512, 3))
        self.b6 = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                nn.Flatten(),
                                nn.Linear(512,num_classes))

    def resnet_block(self, Residual, input_channels, num_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            # 其他模块在第一个残差块里将上一个模块的通道数翻倍，并将高和宽减半
            if i==0 and not first_block:
                blk.append(Residual(input_channels, num_channels, use_lxlconv=True, strides=2))
            # 第一个模块的通道同输入通道数一致;由于之前已使用步幅为2的最大汇聚层，故无须减小高和宽
            else:
                blk.append(Residual(num_channels, num_channels))
        return blk

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        return x

