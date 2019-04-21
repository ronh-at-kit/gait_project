import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import datasets, models, transforms

IMAGE_INPUT_SIZE_W = 176
IMAGE_INPUT_SIZE_H = 250
RGB_CHANNELS = 3

IMAGE_AFTER_CONV_SIZE_W = 3
IMAGE_AFTER_CONV_SIZE_H = 5
CHANNELS_OUT = 6
#
# LR = 0.0001
# MOMENTUM = 0.9

class PretrainConvFlow(nn.Module):
    def __init__(self):
        super(PretrainConvFlow, self).__init__()
        self.avialable_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.conv1 = nn.Conv2d(RGB_CHANNELS, 6, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 16, 3)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(16, 16, 3)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(16, CHANNELS_OUT, 3)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(IMAGE_AFTER_CONV_SIZE_W * IMAGE_AFTER_CONV_SIZE_H * CHANNELS_OUT, 120)
        self.fc2 = nn.Linear(120, 20)
        self.fc3 = nn.Linear(20, 3)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# print("Net defined")
# net = PretrainConvFlow.PretrainConvFlow()


