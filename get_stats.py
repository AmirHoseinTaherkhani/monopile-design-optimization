import numpy as np
import pandas as pd
import os

train_dir = 'train_noise'

train_file_list = os.listdir(train_dir)

output = []
parameters = []
cpt = []
cpt_num = []
for train_file in train_file_list:
    print(train_file)
    data = pd.read_csv(train_dir + '/' + train_file, header=0, sep=' ').values
    output.append(data[[0,1],1])
    parameters.append(data[[2,3,4], 1])
    cpt.extend(data[5:, 1])
    cpt_num.append(len(data[5:, 1]))

output = np.asarray(output)
parameters = np.asarray(parameters)
cpt = np.asarray(cpt)
cpt_num = np.asarray(cpt_num)

print('output: max:', np.log(output).max(axis=0))
print('output: min:', np.log(output).min(axis=0))
print('parameters: max:', parameters.max(axis=0))
print('parameters: min:', parameters.min(axis=0))
print('parameter 2: max:', np.log(parameters[:,1]).max(axis=0))
print('parameter 2: min:', np.log(parameters[:,1]).min(axis=0))
print('cpt: max', cpt.max())
print('cpt: min', cpt.min())
print('max cpt num:', cpt_num.max())

print('output: mean:', output.mean(axis=0))
print('output: std:', output.std(axis=0))
print('parameters: mean:', parameters.mean(axis=0))
print('parameters: std:', parameters.std(axis=0))
print('cpt: mean', cpt.mean())
print('cpt: std', cpt.std())