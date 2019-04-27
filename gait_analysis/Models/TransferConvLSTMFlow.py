import torch
import torch.nn as nn

import torchvision
from torchvision import datasets, models, transforms

from gait_analysis.Models.PretrainConvFlow import PretrainConvFlow

LSTM_INPUT_SIZE = 6*5*3
LSTM_HIDDEN_FEATURES = 40
NR_LSTM_UNITS = 1

TIME_STEPS = 40
BATCH_SIZE = 1

class TransferConvLSTMFlow(nn.Module):
    def __init__(self):
        super(TransferConvLSTMFlow, self).__init__()

        self.conv_net = PretrainConvFlow()
        self.conv_net.load_state_dict(torch.load('/mnt/DATA/HIWI/IBT/saved_models/PretrainConvFlow/001_all_seq_ts40.pt'))
        self.conv_net.eval()
        for param in self.conv_net.parameters():
            param.requires_grad = False

        self.conv_layers = self.conv_net.features
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
            # print("X[i]",x[i].size())
            x_arr[i] = self.conv_layers(x[i])
            # print("X_arr[i]", x[i].size())

        x = x_arr.view(TIME_STEPS, BATCH_SIZE, 6*5*3)
        #         print("x.Size()",x.size())
        x = x.permute(1, 0, 2)
        x, _ = self.lstm(x)
        x = self.classifier(x)

        return x