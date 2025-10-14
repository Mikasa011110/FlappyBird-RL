import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CNN_FeaturesExtract(nn.Module):
    def __init__(self, frames=4):
        super(CNN_FeaturesExtract, self).__init__()
        self.conv1 = nn.Conv2d(frames, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 自动计算 flatten 大小
        with torch.no_grad():
            dummy = torch.zeros(1, 4, 80, 80)
            x = self._forward_features(dummy)
            n_flatten = x.shape[1]

        self.fc1 = nn.Linear(n_flatten, 256)

    def _forward_features(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        return x.view(x.size(0), -1)

    def forward(self, x):
        x = self._forward_features(x)
        x = F.relu(self.fc1(x))
        return x
# class CNN_FeaturesExtract(nn.Module):
#     def __init__(self, frames=4):
#         super(CNN_FeaturesExtract, self).__init__()
#         self.conv1 = nn.Conv2d(frames, 32, kernel_size=8, stride=4)  # 输入是4帧灰度图像 80x80x4，输出32个特征图 20x20x32
#         self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2) # 最大池化层 # 20x20x32-> 10x10x32
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2) # 输入10x10x32，输出64个特征图 64x5x5
#         self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2) # 最大池化层 # 5x5x64-> 3x3x64
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1) # 输入3x3x64，输出64个特征图 64x3x3
#         self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2) # 最大池化层 # 64x3x3-> 64x2x2

#         self.fc1 = nn.Linear(64 * 2 * 2, 256) # 全连接层
#         # self.fc2 = nn.Linear(256, action_space) # 输出层

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.pool_1(x)
#         x = self.conv2(x)
#         x = self.pool_2(x)
#         x = self.conv3(x)
#         x = self.pool_3(x)
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         return x  # 不进行到输出层，只通过3层卷积层，只将4张图片的特征浓缩成64x2x2的特征图，并展平为1x256的特征向量


class Qnet(torch.nn.Module): 
    def __init__(self, frames=4, hidden_dim=128, action_dim=2):
        super(Qnet, self).__init__()
        '''CNN特征提取器'''
        self.cnn_featuresExtract = CNN_FeaturesExtract(frames)
        
        self.fc1 = torch.nn.Linear(256, hidden_dim) # 全连接层，输入为展平的特征向量
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim) # 输出层

    def forward(self, x): 
        x = self.cnn_featuresExtract(x) # 通过CNN特征提取器
        x = x.view(x.size(0), -1)         # 展平
        x = F.relu(self.fc1(x))           # 使用ReLU激活函数
        q_values = self.fc2(x)            # 输出层：输出每个动作的Q值
        return q_values