import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .misc import to_var


def weights_init(m):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        m.weight.data.normal_(1, 0.02)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()


class ConvPool2D(nn.Module):
    def __init__(self, channels, kernel_sizes = 3, paddings = None,
                 batch_norm = True, nonlinear_type = 'RELU', last_nonlinear = False,
                 sampling_type = 'NONE', sampling_sizes = 1):
        super(ConvPool2D, self).__init__()

        # settings
        num_layers = len(channels) - 1
        if not isinstance(kernel_sizes, (list, tuple)):
            kernel_sizes = [kernel_sizes] * num_layers
        if not isinstance(paddings, (list, tuple)):
            paddings = [paddings] * num_layers
        if not isinstance(sampling_sizes, (list, tuple)):
            sampling_sizes = [sampling_sizes] * num_layers

        # sanity checks
        assert nonlinear_type in ['RELU', 'LEAKY-RELU'], 'unsupported nonlinear type "{0}"'.format(nonlinear_type)
        assert sampling_type in ['NONE', 'UP-DECONV', 'UP-NEAREST', 'UP-BILINEAR', 'SUB-CONV', 'SUB-AVGPOOL',
                                 'SUB-MAXPOOL'], 'unsupported sampling type "{0}"'.format(sampling_type)

        # modules
        modules = []
        for k in range(num_layers):
            in_channels, out_channels = channels[k], channels[k + 1]
            kernel_size, padding, sampling_size = kernel_sizes[k], paddings[k], sampling_sizes[k]

            # default padding
            if padding is None:
                padding = (kernel_size - 1) // 2

            # convolution
            if sampling_type == 'UP-DECONV' and sampling_size != 1:
                # fixme
                modules.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size + 1, sampling_size, padding))
            if sampling_type == 'SUB-CONV' and sampling_size != 1:
                modules.append(nn.Conv2d(in_channels, out_channels, kernel_size, sampling_size, padding))
            else:
                modules.append(nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding))

            if last_nonlinear or k + 1 != num_layers:
                # batchnorm
                if batch_norm:
                    modules.append(nn.BatchNorm2d(out_channels))

                # nonlinear
                if nonlinear_type == 'RELU':
                    modules.append(nn.ReLU(True))
                elif nonlinear_type == 'LEAKY-RELU':
                    modules.append(nn.LeakyReLU(0.2, True))

            # sampling
            if sampling_type == 'SUB-AVGPOOL' and sampling_size != 1:
                modules.append(nn.AvgPool2d(sampling_size))
            elif sampling_type == 'SUB-MAXPOOL' and sampling_size != 1:
                modules.append(nn.MaxPool2d(sampling_size))
            elif sampling_type == 'UP-NEAREST' and sampling_size != 1:
                modules.append(nn.Upsample(scale_factor = sampling_size, mode = 'nearest'))
            elif sampling_type == 'UP-BILINEAR' and sampling_size != 1:
                modules.append(nn.Upsample(scale_factor = sampling_size, mode = 'bilinear'))

        # network & initialization
        self.network = nn.Sequential(*modules)
        self.network.apply(weights_init)

    def forward(self, inputs):
        return self.network.forward(inputs)


class ConvPool3D(nn.Module):
    def __init__(self, num_layers, in_channels = 3, out_channels = 8, batch_norm = True):
        super(ConvPool3D, self).__init__()

        # set up number of layers
        if isinstance(num_layers, int):
            num_layers = [num_layers, 0]

        network = []

        # several 3x3 convolutional layers and max-pooling layers
        for k in range(num_layers[0]):
            # 3d convolution
            network.append(nn.Conv3d(in_channels, out_channels, 3, padding = 1))

            # batch normalization
            if batch_norm:
                network.append(nn.BatchNorm3d(out_channels))

            # non-linearity and max-pooling
            network.append(nn.LeakyReLU(0.2, True))
            network.append(nn.MaxPool3d(2))

            # double channel size
            in_channels = out_channels
            out_channels *= 2

        # several 1x1 convolutional layers
        for k in range(num_layers[1]):
            # 3d convolution
            network.append(nn.Conv3d(in_channels, in_channels, 1))

            # batch normalization
            if batch_norm:
                network.append(nn.BatchNorm3d(in_channels))

            # non-linearity
            network.append(nn.LeakyReLU(0.2, True))

        # set up modules for network
        self.network = nn.Sequential(*network)
        self.network.apply(weights_init)

    def forward(self, inputs):
        return self.network.forward(inputs)


class SeqEncoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size = 128, num_layers = 1, dropout = 0):
        super(SeqEncoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # modules
        self.lstm = nn.LSTM(input_size = self.input_size,
                            hidden_size = self.hidden_size,
                            num_layers = self.num_layers,
                            batch_first = True,
                            dropout = self.dropout,
                            bidirectional = True)
        self.linear = nn.Linear(hidden_size * 2, output_size)
        self.apply(weights_init)

    def forward(self, inputs):
        hiddens, _ = self.lstm.forward(inputs, (
            to_var(torch.zeros(self.num_layers * 2, inputs.size(0), self.hidden_size)),
            to_var(torch.zeros(self.num_layers * 2, inputs.size(0), self.hidden_size))
        ))
        hiddens = hiddens[:, -1, :].contiguous()
        outputs = self.linear.forward(hiddens)
        return outputs


class SeqLabeler(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size = 256, num_layers = 1, dropout = 0.9):
        super(SeqLabeler, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # set up modules for recurrent neural networks
        self.rnn = nn.LSTM(input_size = input_size,
                           hidden_size = hidden_size,
                           num_layers = num_layers,
                           batch_first = True,
                           dropout = dropout,
                           bidirectional = True)
        self.rnn.apply(weights_init)

        # set up modules to compute classification
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
        self.classifier.apply(weights_init)

    def forward(self, inputs):
        # set up batch size
        batch_size = inputs.size(0)

        # compute hidden and cell
        hidden = Variable(torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).cuda())
        cell = Variable(torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).cuda())
        hidden_cell = (hidden, cell)

        # recurrent neural networks
        outputs, _ = self.rnn.forward(inputs, hidden_cell)
        outputs = outputs.contiguous().view(-1, self.hidden_size * 2)

        # compute classifications by outputs
        outputs = self.classifier.forward(outputs)
        outputs = F.softmax(outputs)
        outputs = outputs.view(batch_size, -1, self.num_classes)
        return outputs
