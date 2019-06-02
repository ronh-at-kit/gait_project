import torch
import torch.nn as nn

import torchvision
from torchvision import datasets, models, transforms

IMAGE_INPUT_SIZE_W = 176
IMAGE_INPUT_SIZE_H = 250
CHANNELS_IN = 3

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

        self.features = nn.Sequential(
            nn.Conv2d(CHANNELS_IN, 6, 3),
            nn.ReLU(inplace= False),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6, 16, 3),
            nn.ReLU(inplace= False),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 16, 3),
            nn.ReLU(inplace= False),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 16, 3),
            nn.ReLU(inplace= False),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, CHANNELS_OUT, 3),
            nn.ReLU(inplace= False),
            nn.MaxPool2d(2, 2)
            )
        self.classifier = nn.Sequential(
            nn.Linear(IMAGE_AFTER_CONV_SIZE_W * IMAGE_AFTER_CONV_SIZE_H * CHANNELS_OUT, 120),
            nn.ReLU(inplace= False),
            nn.Linear(120, 20),
            nn.ReLU(inplace= False),
            nn.Linear(20, 3)
            )

    def forward(self, x):
        # for i in len(self.features):
        #     x = F.MaxPool2d(F.relu(self.features[i](x)))
        x = self.features(x)
        x = x.view(x.size(0),-1)
        # for i in len(self.classifier)-1:
        #     x = F.relu(self.classifier[i](x))
        # x = self.classifier[-1](x)
        x = self.classifier(x)
        return x

# print("Net defined")
# net = PretrainConvFlow.PretrainConvFlow()


