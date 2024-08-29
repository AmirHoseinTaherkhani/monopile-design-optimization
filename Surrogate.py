import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import Model
from dataset2 import myDataset

import numpy as np
import pandas as pd

from multiprocessing import freeze_support
import matplotlib.pyplot as plt
from Nomarlize import *

plt.style.use('seaborn')


def model_test(L, Ip, h = 29.29282124):

    batch_size = 1
    num_workers = 1
    torch.manual_seed(77)
    torch.cuda.manual_seed(77)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = myDataset('Optimization')
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                             num_workers=num_workers)

    model = Model().to(device)
    model.load_state_dict(torch.load('Finalized_model/models5_800.pkl'))
    model.eval()

    start = time.time()

    name_numpy = []
    prediction1_numpy = []
    prediction2_numpy = []
    output1_numpy = []
    output2_numpy = []
    for i, data in enumerate(test_loader):
        # print(i)
        # output = normalize_output(data['output']).to(device, dtype=torch.float)
        # print(data['parameters'])
        # parameters = normalize_input(data['parameters']).to(device, dtype=torch.float)
        cpt = normalize_cpt_minmax(data['cpt']).to(device, dtype=torch.float)
        params = normalize_input(torch.tensor(np.array([[L, Ip, h]]))).to(device, dtype=torch.float)
        prediction = model(cpt, params)



        prediction_numpy = prediction.cpu().detach().numpy()
    #     output_numpy = output.cpu().detach().numpy()
    #
    #     prediction1_numpy.extend(prediction_numpy[:, 0])
    #     prediction2_numpy.extend(prediction_numpy[:, 1])
    #
    #     output1_numpy.extend(output_numpy[:, 0])
    #     output2_numpy.extend(output_numpy[:, 1])
    #
    #     name_numpy.extend(data['file_names'])
    #     # print(parameters)
    #     # print(torch.tensor(parameters.cpu().detach().numpy()).to(device, dtype=torch.float))
    #     print(prediction)
    return prediction_numpy[0]


if __name__ == '__main__':
    def Ip(D, tw):
        return (1/64)*np.pi*(D**4-(D-2*tw)**4)
    print(model_test(6, Ip(3.05, 0.055), 29.29282124))



