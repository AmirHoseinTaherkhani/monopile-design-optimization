import time
import numpy as np
import pandas as pd
from collections import deque
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import Model
from dataset2 import myDataset

from Nomarlize import *

from multiprocessing import freeze_support



def mean_absolute_relative_error(prediction, output):
    loss = torch.mean(torch.abs((prediction - output) / output))
    return loss


def model_train():
    batch_size = 128
    num_epochs = 1001
    num_workers = 1
    lr = 0.001


    torch.manual_seed(77)
    torch.cuda.manual_seed(77)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load = None #'model_600'

    loss_crit = 'RelativeError'
    optim_crit = 'adam'

    train_dataset = myDataset('train_set')
    val_dataset = myDataset('val_set')
    test_dataset = myDataset('test_set')

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=1,
                            shuffle=False)
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=1,
                            shuffle=False)

    model = Model()

    if load is not None:
        model.load_state_dict(torch.load(f'{load}.pkl'))
        initial_epoch = int(load.split('_')[1])
    else:
        initial_epoch = 0

    model = model.to(device)

    if loss_crit == 'MSELoss':
        criterion = nn.MSELoss()
    elif loss_crit == 'RelativeError':
        criterion = mean_absolute_relative_error

    if optim_crit == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    start = time.time()
    for epoch in range(initial_epoch, num_epochs):


        totalloss = 0
        for i, data in enumerate(train_loader):
            output = normalize_output_power(data['output']).to(device, dtype=torch.float)
            parameters = normalize_input(data['parameters']).to(device, dtype=torch.float)
            cpt = normalize_cpt_minmax(data['cpt']).to(device, dtype=torch.float)

            prediction = model(cpt, parameters)

            loss = criterion(prediction, output) * 1e2

            totalloss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



            if i % 10 == 0:
                print("Epoch:%d;     Iteration:%d;      Loss:%f" % (epoch, i, loss))


        if epoch % 10 == 0:
            torch.save(model.state_dict(),
                       './model_' + str(epoch) + '.pkl')

    end = time.time()
    print((end - start) / 60)

if __name__ == '__main__':
    freeze_support()
    model_train()
