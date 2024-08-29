import torch.nn as nn
import torch
from torch import autograd


class BasicBlock(nn.Module):
    def __init__(self, channel_num):
        super(BasicBlock, self).__init__()

        # TODO: 3x3 convolution -> relu
        # the input and output channel number is channel_num
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(channel_num, channel_num, 3, padding=1),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(channel_num, channel_num, 3, padding=1),
            nn.BatchNorm2d(channel_num),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # TODO: forward
        residual = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x + residual
        out = self.relu(x)
        return out


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        feature_extraction_cpt1_list = []

        feature_extraction_cpt1_list.append(nn.Conv1d(1, 2, kernel_size=3, padding=1))
        feature_extraction_cpt1_list.append(nn.ReLU())
        feature_extraction_cpt1_list.append(nn.Conv1d(2, 2, kernel_size=3, padding=1))
        feature_extraction_cpt1_list.append(nn.ReLU())
        feature_extraction_cpt1_list.append(nn.MaxPool1d(kernel_size=2, stride=2))

        feature_extraction_cpt1_list.append(nn.Conv1d(2, 4, kernel_size=3, padding=1))
        feature_extraction_cpt1_list.append(nn.ReLU())
        feature_extraction_cpt1_list.append(nn.Conv1d(4, 4, kernel_size=3, padding=1))
        feature_extraction_cpt1_list.append(nn.ReLU())
        feature_extraction_cpt1_list.append(nn.MaxPool1d(kernel_size=2, stride=2))

        feature_extraction_cpt1_list.append(nn.Conv1d(4, 8, kernel_size=3, padding=1))
        feature_extraction_cpt1_list.append(nn.ReLU())
        feature_extraction_cpt1_list.append(nn.Conv1d(8, 8, kernel_size=3, padding=1))
        feature_extraction_cpt1_list.append(nn.ReLU())
        feature_extraction_cpt1_list.append(nn.MaxPool1d(kernel_size=2, stride=2))


        #mine
        feature_extraction_cpt1_list.append(nn.Conv1d(8, 16, kernel_size=3, padding=1))
        feature_extraction_cpt1_list.append(nn.ReLU())
        feature_extraction_cpt1_list.append(nn.Conv1d(16, 16, kernel_size=3, padding=1))
        feature_extraction_cpt1_list.append(nn.ReLU())
        feature_extraction_cpt1_list.append(nn.MaxPool1d(kernel_size=2, stride=2))



        feature_extraction_cpt1_list.append(nn.Conv1d(16, 16, kernel_size=3, padding=1))
        feature_extraction_cpt1_list.append(nn.ReLU())
        feature_extraction_cpt1_list.append(nn.Conv1d(16, 16, kernel_size=3, padding=1))
        feature_extraction_cpt1_list.append(nn.ReLU())
        feature_extraction_cpt1_list.append(nn.MaxPool1d(kernel_size=2, stride=2))

        feature_extraction_cpt1_list.append(nn.Conv1d(16, 32, kernel_size=3, padding=1))
        feature_extraction_cpt1_list.append(nn.ReLU())
        feature_extraction_cpt1_list.append(nn.Conv1d(32, 32, kernel_size=3, padding=1))
        feature_extraction_cpt1_list.append(nn.ReLU())
        feature_extraction_cpt1_list.append(nn.MaxPool1d(kernel_size=2, stride=2))

        feature_extraction_cpt1_list.append(nn.Conv1d(32, 32, kernel_size=3, padding=1))
        feature_extraction_cpt1_list.append(nn.ReLU())
        feature_extraction_cpt1_list.append(nn.Conv1d(32, 32, kernel_size=3, padding=1))
        feature_extraction_cpt1_list.append(nn.ReLU())
        feature_extraction_cpt1_list.append(nn.MaxPool1d(kernel_size=2, stride=2))

        feature_extraction_cpt1_list.append(nn.Conv1d(32, 64, kernel_size=3, padding=1))
        feature_extraction_cpt1_list.append(nn.ReLU())
        feature_extraction_cpt1_list.append(nn.Conv1d(64, 64, kernel_size=3, padding=1))
        feature_extraction_cpt1_list.append(nn.ReLU())
        feature_extraction_cpt1_list.append(nn.MaxPool1d(kernel_size=2, stride=2))

        feature_extraction_cpt1_list.append(nn.Conv1d(64, 64, kernel_size=3, padding=1))
        feature_extraction_cpt1_list.append(nn.ReLU())
        feature_extraction_cpt1_list.append(nn.Conv1d(64, 64, kernel_size=3, padding=1))
        feature_extraction_cpt1_list.append(nn.ReLU())
        feature_extraction_cpt1_list.append(nn.MaxPool1d(kernel_size=2, stride=2))

        self.feature_extraction_cpt1 = nn.Sequential(*feature_extraction_cpt1_list)

        feature_extraction_cpt2_list = []

        feature_extraction_cpt2_list.append(nn.Linear(in_features=64, out_features=1024))
        feature_extraction_cpt2_list.append(nn.ReLU())
        feature_extraction_cpt2_list.append(nn.Linear(in_features=1024, out_features=2048))
        feature_extraction_cpt2_list.append(nn.ReLU())
        feature_extraction_cpt2_list.append(nn.Linear(in_features=2048, out_features=512))
        feature_extraction_cpt2_list.append(nn.ReLU())
        feature_extraction_cpt2_list.append(nn.Linear(in_features=512, out_features=3))

        self.feature_extraction_cpt2 = nn.Sequential(*feature_extraction_cpt2_list)

        feature_extraction_parameters_list = []

        feature_extraction_parameters_list.append(nn.Linear(in_features=3, out_features=10))
        feature_extraction_parameters_list.append(nn.ReLU())
        feature_extraction_parameters_list.append(nn.Linear(in_features=10, out_features=20))
        feature_extraction_parameters_list.append(nn.ReLU())
        feature_extraction_parameters_list.append(nn.Linear(in_features=20, out_features=10))
        feature_extraction_parameters_list.append(nn.ReLU())
        feature_extraction_parameters_list.append(nn.Linear(in_features=10, out_features=3))

        self.feature_extraction_parameters = nn.Sequential(*feature_extraction_parameters_list)

        predictor_list = []

        predictor_list.append(nn.Linear(in_features=6, out_features=6))
        predictor_list.append(nn.ReLU())
        # predictor_list.append(nn.Linear(in_features=8, out_features=4))
        # predictor_list.append(nn.ReLU())

        predictor_list.append(nn.Linear(in_features=6, out_features=2))

        self.predictor = nn.Sequential(*predictor_list)

    def forward(self, cpt, parameters):
        features_cpt1 = self.feature_extraction_cpt1(cpt)
        features_cpt1 = features_cpt1.view(features_cpt1.size(0), -1)
        features_cpt2 = self.feature_extraction_cpt2(features_cpt1)

        features_parameters = self.feature_extraction_parameters(parameters)

        features = torch.cat((features_cpt2, features_parameters), 1)

        prediction = self.predictor(features)

        return prediction
