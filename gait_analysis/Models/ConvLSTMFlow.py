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


LSTM_INPUT_SIZE = IMAGE_AFTER_CONV_SIZE_W*IMAGE_AFTER_CONV_SIZE_H*CHANNELS_OUT
LSTM_HIDDEN_FEATURES = 40
NR_LSTM_UNITS = 1

TIME_STEPS = 40
BATCH_SIZE = 1


class ConvLSTMFlow(nn.Module):
    def __init__(self):
        super(ConvLSTMFlow, self).__init__()

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
        self.lstm = nn.LSTM(LSTM_INPUT_SIZE,
                            LSTM_HIDDEN_FEATURES,
                            NR_LSTM_UNITS,
                            batch_first=True)  # horizontal direction
        self.classifier = nn.Sequential(
            nn.Linear(LSTM_HIDDEN_FEATURES, 20),
            #             nn.ReLU(),
            #             nn.Linear(120,20),
            nn.ReLU(),
            nn.Linear(20, 3)
        )

        # self.hidden = self.init_hidden()
        print("TODO: doesn't x = x.view(BATCH_SIZE,TIME_STEPS*LSTM_HIDDEN_FEATURES) mix up batch size order?")
        print("TODO: BATCH_SIZE is currently fixed and one")
        print("TODO: Is x_arr in device?")

    #
    # def init_hidden(self):
    #     return (torch.randn(NR_LSTM_UNITS, BATCH_SIZE, LSTM_HIDDEN_FEATURES),
    #             torch.randn(NR_LSTM_UNITS, BATCH_SIZE, LSTM_HIDDEN_FEATURES))

    def forward(self, x):
        # print("X[0]:",x[0].size())
        x_arr = torch.zeros(TIME_STEPS, BATCH_SIZE, 6, 5, 3)# .to(self.avialable_device)

        for i in range(TIME_STEPS):
            x_arr[i] = self.features(x[i])
            # print("X_arr[i]", x[i].size())

        x = x_arr.view(TIME_STEPS, BATCH_SIZE, 6*5*3)
        #         print("x.Size()",x.size())
        x = x.permute(1, 0, 2)
        x, _ = self.lstm(x)
        x = self.classifier(x)

        return x