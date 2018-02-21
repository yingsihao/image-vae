from __future__ import print_function

import argparse
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
from tqdm import tqdm
from PIL import Image
import numpy as np

from hparams import hp
from misc import visualize
from utils import set_cuda_devices
from utils.shell import mkdir
from utils.torch import Logger, load_snapshot, save_snapshot, to_np, to_var

import matplotlib.pyplot as plt
import scipy.io as sio

def to_targets(inputs):
    stop_tokens = torch.stack([to_var(hp.stop_token).unsqueeze(0)] * inputs.size(0))
    targets = torch.cat([inputs, stop_tokens], 1)
    return targets


def to_xc(inputs):
    return inputs[:, :, :2].contiguous(), \
           inputs[:, :, -3:].contiguous()


def r_loss(outputs, targets, lengths):
    batch_size = targets.size(0)

    x1, c1 = to_xc(outputs)
    x2, c2 = to_xc(targets)

    # mask
    mask = torch.zeros(batch_size, hp.max_length + 1)
    for k, length in enumerate(lengths):
        mask[k, :length] = 1
    mask = to_var(mask)

    # loss
    loss_c = -torch.sum(c2 * torch.log(c1))
    loss_x = torch.sum(mask * torch.sum(torch.abs(x1 - x2), 2))
    loss = (loss_c + loss_x) / (batch_size * (hp.max_length + 1))
    return loss

class VAE(nn.Module):
    def __init__(self, nc, ngf, ndf, latent_variable_size):
        super(VAE, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size

        # encoder
        self.e1 = nn.Conv2d(nc, ndf, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(ndf)

        self.e2 = nn.Conv2d(ndf, ndf*2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ndf*2)

        self.e3 = nn.Conv2d(ndf*2, ndf*4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ndf*4)

        self.e4 = nn.Conv2d(ndf*4, ndf*8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(ndf*8)

        self.e5 = nn.Conv2d(ndf*8, ndf*8, 4, 2, 1)
        self.bn5 = nn.BatchNorm2d(ndf*8)

        self.fc1 = nn.Linear(ndf*8*4*4, latent_variable_size)
        self.fc2 = nn.Linear(ndf*8*4*4, latent_variable_size)

        # decoder
        self.d1 = nn.Linear(latent_variable_size, ngf*8*2*4*4)

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd1 = nn.ReplicationPad2d(1)
        self.d2 = nn.Conv2d(ngf*8*2, ngf*8, 3, 1)
        self.bn6 = nn.BatchNorm2d(ngf*8, 1.e-3)

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd2 = nn.ReplicationPad2d(1)
        self.d3 = nn.Conv2d(ngf*8, ngf*4, 3, 1)
        self.bn7 = nn.BatchNorm2d(ngf*4, 1.e-3)

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd3 = nn.ReplicationPad2d(1)
        self.d4 = nn.Conv2d(ngf*4, ngf*2, 3, 1)
        self.bn8 = nn.BatchNorm2d(ngf*2, 1.e-3)

        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd4 = nn.ReplicationPad2d(1)
        self.d5 = nn.Conv2d(ngf*2, ngf, 3, 1)
        self.bn9 = nn.BatchNorm2d(ngf, 1.e-3)

        self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd5 = nn.ReplicationPad2d(1)
        self.d6 = nn.Conv2d(ngf, nc, 3, 1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        h5 = h5.view(-1, self.ndf*8*4*4)

        return self.fc1(h5), self.fc2(h5)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if args.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, self.ngf*8*2, 4, 4)
        h2 = self.leakyrelu(self.bn6(self.d2(self.pd1(self.up1(h1)))))
        h3 = self.leakyrelu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
        h4 = self.leakyrelu(self.bn8(self.d4(self.pd3(self.up3(h3)))))
        h5 = self.leakyrelu(self.bn9(self.d5(self.pd4(self.up4(h4)))))

        return self.sigmoid(self.d6(self.pd5(self.up5(h5))))

    def get_latent_var(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        z = self.reparametrize(mu, logvar)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar

reconstruction_function = nn.BCELoss()
reconstruction_function.size_average = False

def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)
    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return BCE + KLD

if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument('--exp', default = 'default')
    parser.add_argument('--resume', default = None)
    parser.add_argument('--gpu', default = '3')
    parser.add_argument('--cuda', default = True)

    # dataset
    parser.add_argument('--data_path', default = './data')
    parser.add_argument('--categories', default = None)
    parser.add_argument('--workers', default = 8, type = int)
    parser.add_argument('--batch', default = 128, type = int)

    # training
    parser.add_argument('--epochs', default = 100, type = int)
    parser.add_argument('--snapshot', default = 16, type = int)
    parser.add_argument('--learning_rate', default = 0.0001, type = float)

    # arguments
    args = parser.parse_args()
    print('==> arguments parsed')
    for key in vars(args):
        print('[{0}] = {1}'.format(key, getattr(args, key)))

    # cuda devices
    set_cuda_devices(args.gpu)

    # datasets & loaders
    '''
    data, loaders = {}, {}
    for split in ['train', 'test']:
        data[split] = QuickDraw(data_path = args.data_path, categories = categories, split = split)
        loaders[split] = DataLoader(data[split], batch_size = args.batch, shuffle = True, num_workers = args.workers)
    print('==> dataset loaded')
    print('[size] = {0} + {1}'.format(len(data['train']), len(data['test'])))
    '''

    data, loaders = {}, {}
    imgs = {'train':[], 'test':[]}
    train_dir = os.path.join(args.data_path, 'train')
    test_dir = os.path.join(args.data_path, 'test')
    n_train = 45032
    n_test = 5006
    for i in tqdm(range(n_train), desc = 'load train data'):
        img = Image.open(train_dir + '/{}.jpg'.format(str(i)))
        if not (np.array(img).shape == (136, 136, 3)):
            print('wrong!!!')
        else:
            imgs['train'].append(np.array(img)[4:132, 4:132])
    for i in tqdm(range(n_test), desc = 'load test data'):
        img = Image.open(test_dir + '/{}.jpg'.format(str(i)))
        if not (np.array(img).shape == (136, 136, 3)):
            print('wrong!!!')
        else:
            imgs['test'].append(np.array(img)[4:132, 4:132])
    for split in ['train', 'test']:
        tmp = np.array(imgs[split])
        #print(tmp.shape)
        tmp = torch.from_numpy(tmp)
        tmp = tmp.permute(0, 3, 1, 2)
        print(tmp.shape)
        data[split] = TensorDataset(data_tensor = tmp, target_tensor = tmp)
        loaders[split] = DataLoader(data[split], batch_size = args.batch, shuffle = True, num_workers = args.workers)
    print('==> dataset loaded')

    # model
    model = VAE(nc=3, ngf=128, ndf=128, latent_variable_size=500).cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    # experiment path
    exp_path = os.path.join('exp', args.exp)
    mkdir(exp_path, clean = False)

    # logger
    logger = Logger(exp_path)

    # load snapshot
    if args.resume is not None:
        epoch = load_snapshot(args.resume, model = model, optimizer = optimizer, returns = 'epoch')
        print('==> snapshot "{0}" loaded'.format(args.resume))
    else:
        epoch = 0

    # iterations
    for epoch in range(epoch, args.epochs):
        step = epoch * len(data['train'])
        print('==> epoch {0} (starting from step {1})'.format(epoch + 1, step + 1))

        # training
        model.train()
        for inputs, lengths in tqdm(loaders['train'], desc = 'train'):
            inputs = to_var(inputs)

            # forward
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(inputs)

            # reconstruction & kl divergence loss
            loss = loss_function(recon_batch, inputs, mu, logvar)

            # logger
            logger.scalar_summary('train-loss', loss.data[0], step)
            step += inputs.size(0)

            # backward
            loss.backward()
            optimizer.step()

        # testing
        model.eval()

        test_loss = 0
        for inputs, lengths in tqdm(loaders['test'], desc = 'test'):
            inputs = to_var(inputs, volatile = True)

            # forward
            recon_batch, mu, logvar = model(inputs)

            torchvision.utils.save_image(inputs.data, './resimgs/Epoch_{}_data.jpg'.format(epoch), nrow=8, padding=2)
            torchvision.utils.save_image(recon_batch.data, './resimgs/Epoch_{}_recon.jpg'.format(epoch), nrow=8, padding=2)

            # reconstruction & kl divergence loss
            test_loss += loss_function(recon_batch, inputs, mu, logvar).data[0]
            # loss_kl += kld_loss(mean, log_var) * targets.size(0) / len(data['test'])

        test_loss /= (len(loaders['test']) * 128)
        logger.scalar_summary('test-loss-r', test_loss, step)
        # logger.scalar_summary('test-loss-kl', loss_kl.data[0], step)

'''
        # visualization
        for split in ['train', 'test']:
            inputs, lengths = iter(loaders[split]).next()
            inputs = to_var(inputs, volatile = True)
            targets = to_targets(inputs)

            # forward
            outputs = model.forward(inputs)

            # visualize
            outputs = [visualize(output) for output in to_np(outputs)]
            targets = [visualize(target) for target in to_np(targets)]

            # logger
            logger.image_summary('{0}-outputs'.format(split), outputs, step)
            logger.image_summary('{0}-targets'.format(split), targets, step)

        # snapshot
        save_snapshot(os.path.join(exp_path, 'latest.pth'),
                      model = model, optimizer = optimizer, epoch = epoch + 1)

        if args.snapshot != 0 and (epoch + 1) % args.snapshot == 0:
            save_snapshot(os.path.join(exp_path, 'epoch-{0}.pth'.format(epoch + 1)),
                          model = model, optimizer = optimizer, epoch = epoch + 1)
'''
