from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

from hparams import hp
from utils.torch import SeqEncoder, to_np, to_var, weights_init


class CEstimator(nn.Module):
    def __init__(self, input_size, output_size):
        super(CEstimator, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # modules
        self.network = nn.Linear(input_size, output_size)
        self.apply(weights_init)

    def forward(self, inputs):
        outputs = self.network.forward(inputs)
        return F.softmax(outputs, dim = 1)


class XEstimator(nn.Module):
    def __init__(self, input_size, output_size):
        super(XEstimator, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # modules
        self.network = nn.Linear(input_size, output_size)
        self.apply(weights_init)

    def forward(self, inputs):
        outputs = self.network.forward(inputs)
        return F.tanh(outputs)


class Decoder(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # modules
        self.linear = nn.Linear(feature_size, hidden_size * 2)
        self.lstm = nn.LSTM(input_size = feature_size + 5,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            batch_first = True,
                            dropout = dropout)
        self.x_estimator = XEstimator(input_size = hidden_size, output_size = 2)
        self.c_estimator = CEstimator(input_size = hidden_size, output_size = 3)
        self.apply(weights_init)

    def forward(self, features, labels = None):
        batch_size = features.size(0)
        start_tokens = torch.stack([to_var(hp.start_token)] * batch_size)

        # hidden & cell
        hidden, cell = torch.split(F.tanh(self.linear(features)), self.hidden_size, 1)
        hidden = torch.stack([hidden.contiguous()] * self.num_layers)
        cell = torch.stack([cell.contiguous()] * self.num_layers)

        if labels is not None:
            inputs = torch.cat([start_tokens.unsqueeze(1), labels], 1)
            inputs = torch.cat([torch.stack([features] * (hp.max_length + 1), 1), inputs], 2)

            outputs, (hiddens, cells) = self.lstm.forward(inputs, (hidden, cell))
            outputs = outputs.contiguous().view(-1, self.hidden_size)

            x = self.x_estimator.forward(outputs).view(-1, hp.max_length + 1, 2)
            c = self.c_estimator.forward(outputs).view(-1, hp.max_length + 1, 3)

            # output
            outputs = torch.cat([x, c], 2)
        else:
            outputs, hiddens, cells = [], [], []

            output = start_tokens
            for k in range(hp.max_length + 1):
                input = torch.cat([features.unsqueeze(1), output.unsqueeze(1)], 2)

                output, (hidden, cell) = self.lstm.forward(input, (hidden, cell))
                output = output.contiguous().squeeze(1)

                x = self.x_estimator.forward(output).view(-1, 2)
                c = self.c_estimator.forward(output).view(-1, 3)

                # sample
                c = to_np(c)
                indices = np.argmax(c, 1)

                c = np.zeros_like(c)
                for i, index in enumerate(indices):
                    c[i, index] = 1
                c = to_var(c)

                # output
                output = torch.cat([x, c], 1)

                # save
                outputs.append(output)
                hiddens.append(hidden.squeeze(0))
                cells.append(cell.squeeze(0))

            # stack
            outputs = torch.stack(outputs, 1)
            hiddens = torch.stack(hiddens, 1)
            cells = torch.stack(cells, 1)

        return outputs, (hiddens, cells)


class SketchVAE(nn.Module):
    def __init__(self, feature_size = 128, hidden_size = 512, num_layers = 4, dropout = 0):
        super(SketchVAE, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # modules
        self.encoder = SeqEncoder(
            input_size = 5,
            output_size = self.feature_size,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            dropout = self.dropout
        )
        self.decoder = Decoder(
            feature_size = self.feature_size,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            dropout = self.dropout
        )
        self.apply(weights_init)

    def forward(self, inputs = None, returns = None):
        if inputs is None:
            z = None
        else:
            z = self.encoder.forward(inputs)

        if self.training:
            outputs, (hiddens, cells) = self.decoder.forward(z, labels = inputs)
        else:
            outputs, (hiddens, cells) = self.decoder.forward(z)

        # returns
        if returns is not None:
            for i, k in enumerate(returns):
                returns[i] = locals()[k]
            return outputs, returns[0] if len(returns) == 1 else returns

        return outputs

class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf = 64,
                 norm_layer = nn.BatchNorm2d, use_dropout = False, gpu_ids = [1]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc = None, submodule = None, norm_layer=norm_layer, innermost = True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc = None, submodule = unet_block, norm_layer = norm_layer, use_dropout = use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc = None, submodule = unet_block, norm_layer = norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc = None, submodule = unet_block, norm_layer = norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc = None, submodule = unet_block, norm_layer = norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc = input_nc, submodule = unet_block, outermost = True, norm_layer = norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc = None,
                 submodule = None, outermost = False, innermost = False, norm_layer = nn.BatchNorm2d, use_dropout = False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size = 4,
                             stride = 2, padding = 1, bias = use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size = 4, stride = 2,
                                        padding = 1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size = 4, stride = 2,
                                        padding = 1, bias = use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size = 4, stride = 2,
                                        padding = 1, bias = use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

