from __future__ import print_function

import argparse
import os

import numpy as np
import scipy.misc
from tqdm import trange

from misc import visualize
from networks import SketchVAE
from utils import set_cuda_devices
from utils.shell import mkdir
from utils.torch import load_snapshot

if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument('--exp', default = 'default')
    parser.add_argument('--resume', default = None)
    parser.add_argument('--gpu', default = '0')

    # demo
    parser.add_argument('--samples', default = 1024, type = int)
    parser.add_argument('--batch', default = 128, type = int)

    # arguments
    args = parser.parse_args()
    print('==> arguments parsed')
    for key in vars(args):
        print('[{0}] = {1}'.format(key, getattr(args, key)))

    # cuda devices
    set_cuda_devices(args.gpu)

    # model
    model = SketchVAE().cuda()

    # demo path
    demo_path = os.path.join('exp', args.exp, 'demo')
    mkdir(demo_path, clean = True)

    # load snapshot
    if os.path.isfile(args.resume):
        epoch = load_snapshot(args.resume, model = model, returns = 'epoch')
        print('==> snapshot "{0}" loaded (epoch {1})'.format(args.resume, epoch))
    else:
        raise FileNotFoundError('no snapshot found at "{0}"'.format(args.resume))

    model.train(False)
    for k in trange(args.samples // args.batch):
        outputs = model.forward(batch_size = args.batch)
        outputs = outputs.data.cpu().numpy()

        for i, output in enumerate(outputs):
            index = k * args.batch + i
            scipy.misc.imsave(os.path.join(demo_path, '{:08d}.png'.format(index)), visualize(output))
            np.save(os.path.join(demo_path, '{:08d}.npy'.format(index)), output)

    # visualization
    with open(os.path.join(demo_path, 'index.html'), 'w') as fp:
        print('<table border="1" style="table-layout: fixed;">', file = fp)
        for k in range(0, args.samples, 8):
            print('<tr>', file = fp)
            for index in range(k, k + 8):
                print('<td halign="center" style="word-wrap: break-word;" valign="top">', file = fp)
                print('<img src="{0}" style="width:128px;">'.format('{:08d}.png'.format(index)), file = fp)
                print('</td>', file = fp)
            print('</tr>', file = fp)
        print('</table>', file = fp)
