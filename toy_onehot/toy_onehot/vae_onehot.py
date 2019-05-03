# this is the traditional vae model

import os
import sys
import numpy as np
import torch
import random
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from itertools import chain
import argparse
from matplotlib import pyplot as plt


def binary_cross_entropy(pred, target, size_average=True):
    pred = pred.view(pred.shape[0], -1)
    target = target.view(target.shape[0], -1)
    loss = -torch.sum( target * torch.log( pred + 1e-20 ) + (1.0 - target) * torch.log( 1.0 - pred + 1e-20 ) )
    if size_average:
        return loss / pred.size()[0]
    else:
        return loss


def loss_function(recon_x, x, mu, logvar):
    BCE = binary_cross_entropy(recon_x, x, size_average=False)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return (BCE + KLD) / x.shape[0]


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 4)

    def forward(self, z):
        h1 = F.elu(self.fc1(z))
        h2 = F.elu(self.fc2(h1))
        h3 = F.elu(self.fc3(h2))

        return F.sigmoid(self.fc4(h3))


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)

        self.fc4 = nn.Linear(64, 2)
        self.fc5 = nn.Linear(64, 2)

    def encode(self, x):
        h1 = F.elu(self.fc1(x))
        h2 = F.elu(self.fc2(h1))
        h3 = F.elu(self.fc3(h2))

        return self.fc4(h3), self.fc5(h3)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


def create_scatter(x_test_list, encoder, savepath=None):
    fig = plt.figure(figsize=(4, 4), facecolor='w')
    ax = fig.add_axes([0, 0, 1, 1])

    for i in range(4):
    # for i in [2, 1, 0, 3]:
        z_out = encoder(x_test_list[i]).data.cpu().numpy()
        plt.scatter(z_out[:, 0], z_out[:, 1],  edgecolor='none', alpha=0.5)

    ax.axis('off')
    fig.savefig(savepath)


parser = argparse.ArgumentParser()

# global parameters
parser.add_argument('--save_dir', default='results_vae')
parser.add_argument('-seed', default=0, type=int, help='random seed')
parser.add_argument('-ctx', default='gpu', help='cpu/gpu')
parser.add_argument('-learning_rate', default=2e-5, type=float, help='learning rate')
parser.add_argument('-batch_size', default=500, type=int, help='batchsize')

if sys.argv[1:] == ['0', '0']:
    args = parser.parse_args([])   # run in pycharm console
else:
    args = parser.parse_args()  # run in cmd


if __name__ == '__main__':
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    encoder = Encoder()
    decoder = Decoder()
        
    if args.ctx == 'gpu':
        encoder.cuda()
        decoder.cuda()
        torch.cuda.manual_seed(args.seed)

    optimizer = optim.Adam(chain(encoder.parameters(), decoder.parameters()), lr=args.learning_rate)
    
    x_test_list = []
    for i in range(4):
        x_test_labels = torch.LongTensor([i] * args.batch_size).view(-1)
        x_test = torch.zeros(x_test_labels.shape[0], 4)
        x_test.scatter_(1, x_test_labels.view(-1, 1), 1)
        if args.ctx == 'gpu':
            x_test = x_test.cuda()
        x_test_list.append(x_test)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    pbar = tqdm(range(80001))
    for iter in pbar: 
        encoder.train()
        idx = torch.LongTensor(args.batch_size).random_(0, 4)
        data = torch.zeros(args.batch_size, 4)
        data.scatter_(1, idx.view(-1, 1), 1)

        if args.ctx == 'gpu':
            data = data.cuda()

        optimizer.zero_grad()
        z, mu, logvar = encoder(data)
        recon_batch = decoder(z)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        optimizer.step()

        pbar.set_description('minibatch loss: %.4f' % loss.item())

        if iter % 5000 == 0:
            encoder_func = lambda x: encoder(x)[0]
            create_scatter(x_test_list, encoder_func, savepath=os.path.join(args.save_dir, '%06d.png' % iter))