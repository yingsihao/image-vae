import os
import pickle

import numpy as np
from torch.utils.data import Dataset

from hparams import hp
from utils.shell import mkdir


def truncate_data(inputs):
    outputs = []
    for data in inputs:
        if len(data) < 16 or len(data) > hp.max_length:
            continue
        data = np.minimum(data, 1024)
        data = np.maximum(data, -1024)
        outputs.append(np.float32(data))
    return outputs


def rescale_data(inputs):
    outputs = []
    for data in inputs:
        data[:, :2] /= np.max(np.abs(data[:, :2]))
        outputs.append(data)
    return outputs


def load_data(data_path, category, split):
    mkdir(data_path, clean = False)
    data_path = os.path.join(data_path, '{0}-{1}.pkl'.format(category, split))

    if not os.path.exists(data_path):
        local_path = os.path.join('/tmp/', '{0}.npz'.format(category))
        remote_path = os.path.join('https://storage.googleapis.com/quickdraw_dataset/sketchrnn/',
                                   '{0}.full.npz'.format(category))
        os.system('wget {0} -O {1}'.format(remote_path, local_path))

        # load
        data = np.load(local_path, encoding = 'latin1')[split]
        data = truncate_data(data)
        data = rescale_data(data)

        # save
        pickle.dump(data, open(data_path, 'wb'))

    return pickle.load(open(data_path, 'rb'))


class QuickDraw(Dataset):
    def __init__(self, data_path, categories, split):
        self.data = np.concatenate([load_data(data_path, c, split) for c in categories])

    def __getitem__(self, index):
        data = self.data[index]
        length = len(data[:, 0])

        # input
        input = np.zeros((hp.max_length, 5))
        input[:length, :2] = data[:, :2]
        input[:length - 1, 2] = 1 - data[:-1, 2]
        input[:length, 3] = data[:, 2]
        input[(length - 1):, 4] = 1
        input[length - 1, 2:4] = 0

        return input, length

    def __len__(self):
        return len(self.data)
