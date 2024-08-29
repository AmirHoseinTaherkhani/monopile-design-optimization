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


def model_test():
    batch_size = 1
    num_workers = 0
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
        print(i)
        output = normalize_output(data['output']).to(device, dtype=torch.float)
        parameters = normalize_input(data['parameters']).to(device, dtype=torch.float)
        cpt = normalize_cpt_minmax(data['cpt']).to(device, dtype=torch.float)

        prediction = model(cpt, parameters)



        prediction_numpy = prediction.cpu().detach().numpy()
        output_numpy = output.cpu().detach().numpy()

        prediction1_numpy.extend(prediction_numpy[:, 0])
        prediction2_numpy.extend(prediction_numpy[:, 1])

        output1_numpy.extend(output_numpy[:, 0])
        output2_numpy.extend(output_numpy[:, 1])

        name_numpy.extend(data['file_names'])
        # print(parameters)
        # print(torch.tensor(parameters.cpu().detach().numpy()).to(device, dtype=torch.float))
        print(prediction)
    return prediction_numpy[0]


    end = time.time()
    # print((end - start) / 60)




    prediction1_numpy = np.array(prediction1_numpy)
    prediction2_numpy = np.array(prediction2_numpy)
    output1_numpy = np.array(output1_numpy)
    output2_numpy = np.array(output2_numpy)



    # plt.scatter(output1_numpy, prediction1_numpy, alpha=.2)
    # plt.plot([0, 10000], [0, 10000], '--', c = 'tab:red')
    # plt.xlim([0, 10000])
    # plt.ylim([0, 10000])
    # plt.xlabel('Ground truth')
    # plt.ylabel('Prediction')
    # plt.title('Output 1')
    # plt.show()
    #
    # plt.scatter(output2_numpy, prediction2_numpy, alpha=.2)
    # plt.plot([0, 20000], [0, 20000], '--', c = 'tab:red')
    # plt.xlim([0, 20000])
    # plt.ylim([0, 20000])
    # plt.xlabel('Ground truth')
    # plt.ylabel('Prediction')
    # plt.title('Output 2')
    # plt.show()
    #
    # plt.scatter(output1_numpy, (prediction1_numpy - output1_numpy) / output1_numpy, alpha=.2)
    # plt.xlim([output1_numpy.min() - 5, output1_numpy.max() + 5])
    # plt.ylim([-1, 1])
    # plt.xlabel('Ground truth')
    # plt.ylabel('Relative error')
    # plt.grid('both')
    # plt.title('Output 1 Relative Error')
    # plt.show()
    # print("Output 1 Average: ", np.mean(np.abs((prediction1_numpy - output1_numpy) / output1_numpy)))
    #
    #
    # plt.scatter(output2_numpy, (prediction2_numpy - output2_numpy) / output2_numpy, alpha=.2)
    # plt.xlim([output2_numpy.min() - 5, output2_numpy.max() + 5])
    # plt.ylim([-1, 1])
    # plt.grid('both')
    # plt.xlabel('Ground truth')
    # plt.ylabel('Relative error')
    # plt.title('Output 2 Relative Error')
    # plt.show()
    # print("Output 2 Average: ", np.mean(np.abs((prediction2_numpy - output2_numpy) / output2_numpy)))
    #
    # # pd.DataFrame({'output1' : output1_numpy, 'prediction1': prediction1_numpy, 'output2' : output2_numpy,\
    # #               'prediction2': prediction2_numpy}).to_csv('Updated2/original.csv')
    #

    # print('MSE Loss : ', np.square(prediction_numpy - output_numpy).mean())

# if __name__ == '__main__':
#     # freeze_support()
model_test()


