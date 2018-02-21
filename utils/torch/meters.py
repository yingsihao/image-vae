import math

import numpy as np
import torch

from .misc import to_np


class MSEMeter:
    def __init__(self, root = False):
        self.root = root
        self.reset()

    def reset(self):
        self.size = 0
        self.sum = 0.

    def add(self, outputs, targets):
        outputs = to_np(outputs.squeeze())
        targets = to_np(targets.squeeze())
        self.size += outputs.shape[0]
        self.sum += np.sum((outputs - targets) ** 2)

    def value(self):
        value = self.sum / max(self.size, 1)
        return math.sqrt(value) if self.root else value


class ClassErrorMeter:
    def __init__(self, k = [1]):
        self.k = np.sort(k)
        self.reset()

    def reset(self):
        self.size = 0
        self.corrects = {k: 0 for k in self.k}

    def add(self, outputs, targets):
        outputs = to_np(outputs.squeeze())
        targets = to_np(targets.squeeze())

        if np.ndim(targets) == 2:
            targets = np.argmax(targets, 1)

        assert np.ndim(outputs) == 2, 'wrong output size (2D expected)'
        assert np.ndim(targets) == 1, 'wrong target size (1D or 2D expected)'
        assert targets.shape[0] == outputs.shape[0], 'number of outputs and targets do not match'

        predict = torch.from_numpy(outputs).topk(int(self.k[-1]), 1, True, True)[1].numpy()
        correct = (predict == targets[:, np.newaxis].repeat(predict.shape[1], 1))

        self.size += targets.shape[0]
        for k in self.k:
            self.corrects[k] += correct[:, :k].sum()

    def value(self, k = None):
        assert k is None or k in self.k, 'invalid k (this k was not provided at construction time)'

        if k is not None:
            return float(self.corrects[k]) / self.size * 100.
        else:
            values = [self.value(k) for k in self.k]
            return values[0] if len(values) == 1 else values


class ConfusionMeter:
    def __init__(self, num_classes):
        self.confusion = np.ndarray((num_classes, num_classes), dtype = np.int32)
        self.reset()

    def reset(self):
        self.confusion.fill(0)

    def add(self, outputs, targets):
        outputs = to_np(outputs.squeeze())
        targets = to_np(targets.squeeze())

        if np.ndim(outputs) == 2:
            outputs = np.argmax(outputs, 1)
        if np.ndim(targets) == 2:
            targets = np.argmax(targets, 1)

        assert outputs.shape[0] == targets.shape[0], 'number of targets and outputs do not match'

        for output, target in zip(outputs, targets):
            self.confusion[target][output] += 1

    def value(self, normalize = True):
        value = self.confusion.astype(np.float32)
        if normalize:
            value /= value.sum(1).clip(min = 1e-12)[:, None]
        return value
