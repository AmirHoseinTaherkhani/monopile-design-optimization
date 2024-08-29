import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


#
class myDataset(Dataset):
    def __init__(self, root):
        Dataset.__init__(self)
        self.data_dir = root
        self.data = os.listdir(self.data_dir)

    def __getitem__(self, index):
        file_name = self.data[index]

        original_data = pd.read_csv(self.data_dir + '/' + file_name, header=0, sep=',').drop('Unnamed: 0', axis = 1).values
        padded_length = 810 - original_data[5:, 1].shape[0]
        output = original_data[:2,0:1].T[0]
        parameters = original_data[2:5,0:1].T[0]
        cpt = original_data[5:, 1]

        #parameters = (parameters - np.array([6.064442, 0.061516, 15.000062])) / np.array(\
        # [149.322788 - 6.064442, 90.306199 - 0.061516, 29.999967 - 15.000062])
        cpt = (cpt - 1.6731150000000001) / (66.071475 - 1.6731150000000001)

        data = {'output': output, 'parameters': parameters,
                'cpt': np.pad(cpt, (0, padded_length), mode='constant').reshape((1, -1)), 'file_names': file_name}

        return data

    def __len__(self):
        return len(self.data)
