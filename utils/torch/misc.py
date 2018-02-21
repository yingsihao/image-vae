import logging
import os

import numpy as np
import torch
from torch.autograd import Variable

thisfile = os.path.abspath(__file__)


def to_np(inputs):
    if isinstance(inputs, Variable):
        inputs = inputs.data

    if torch.is_tensor(inputs):
        inputs = inputs.cpu().numpy()
    return inputs


def to_var(inputs, type = 'float', cuda = True, volatile = False):
    if isinstance(inputs, list):
        for k, input in enumerate(inputs):
            inputs[k] = to_var(input, type, cuda, volatile)
    else:
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs)

        if torch.is_tensor(inputs):
            if cuda:
                inputs = inputs.cuda()
            inputs = Variable(inputs, volatile = volatile)

        if hasattr(inputs, type):
            inputs = getattr(inputs, type)()
    return inputs


def load_snapshot(path, model = None, optimizer = None, returns = None):
    if not os.path.isfile(path):
        raise FileNotFoundError('no snapshot found at "{0}"'.format(path))

    # load snapshot
    snapshot = torch.load(path)

    if model is not None:
        model.load_state_dict(snapshot['model'])
    if optimizer is not None:
        optimizer.load_state_dict(snapshot['optimizer'])

    # returns
    if returns is not None:
        if not isinstance(returns, (list, tuple)):
            returns = [returns]

        returns = [snapshot[k] for k in returns]
        return returns[0] if len(returns) == 1 else returns

    return snapshot


def save_snapshot(path, model = None, optimizer = None, **kwargs):
    snapshot = {k: v for k, v in kwargs.items()}

    if model is not None:
        snapshot['model'] = model.state_dict()
    if optimizer is not None:
        snapshot['optimizer'] = optimizer.state_dict()

    torch.save(snapshot, path)


def load_state_dict(model, state_dict):
    thisfunc = thisfile + '->load_state_dict()'

    # unusable params
    params = []
    for param, x in model.state_dict().items():
        if param in state_dict:
            y = state_dict[param]
            if hasattr(x, 'size') and hasattr(y, 'size') and x.size() != y.size():
                params.append(param)
                state_dict[param] = x
        else:
            params.append(param)
            state_dict[param] = x

    if len(params) > 0:
        logging.warning('{0}: replacing params {1} by initial values'.format(thisfunc, params))

    # redundant params
    params = []
    for param in state_dict.keys():
        if param not in model.state_dict():
            params.append(param)

    for param in params:
        del state_dict[param]

    if len(params) > 0:
        logging.warning('{0}: removing params {1}'.format(thisfunc, params))

    # sanity checks
    assert len([param for param in model.state_dict().keys() if param not in state_dict.keys()]) == 0
    assert len([param for param in state_dict.keys() if param not in model.state_dict().keys()]) == 0

    # state dicts
    model.load_state_dict(state_dict)
