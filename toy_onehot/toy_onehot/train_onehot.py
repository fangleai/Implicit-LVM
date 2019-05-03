
import os
import sys
from utils_onehot import create_scatter, optimizer_step
from models_onehot import Decoder, Nu, Encoder
import torch
from itertools import chain
from torch import optim
import argparse
from torch.autograd import grad
import logging
from tqdm import tqdm

parser = argparse.ArgumentParser()

# global parameters
parser.add_argument('--results_folder', default='results')
parser.add_argument('--train_from', default='')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--test', action="store_true")
parser.add_argument('--log_prefix', default='')

# global training parameters
parser.add_argument('--num_epochs', default=80000, type=int)
parser.add_argument('--num_particles', default=64, type=int)
parser.add_argument('--test_num_particles', default=500, type=int)
parser.add_argument('--num_nu_updates', default=5, type=int)  # converge nu faster and better to avoid NA

# use GPU
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--no_gpu', action="store_true")
parser.add_argument('--plot_freq', type=int, default=5000)

# model and optimizer parameters
parser.add_argument('--nu_lr', default=1e-5, type=float)
parser.add_argument('--end2end_lr', default=1e-5, type=float)

if sys.argv[1:] == ['0', '0']:
    args = parser.parse_args([])   # run in pycharm console
else:
    args = parser.parse_args()  # run in cmd

# training starts
torch.manual_seed(args.seed)
if not os.path.exists(args.results_folder):
    os.makedirs(args.results_folder)
logging.basicConfig(filename=os.path.join(args.results_folder, args.log_prefix + 'eval.log'),
                    level=logging.INFO, format='%(asctime)s--- %(message)s')
if not torch.cuda.is_available(): args.no_gpu = True
gpu = not args.no_gpu
if gpu: torch.cuda.set_device(args.gpu)

x_test = torch.zeros((4 * args.test_num_particles, 4), device='cuda' if gpu else 'cpu')
idx_test = torch.LongTensor([j for i in range(4) for j in [i] * args.test_num_particles]).to(x_test.device)
x_test.scatter_(1, idx_test.view(-1, 1), 1)
eps_test = torch.zeros((4 * args.test_num_particles, 2), device=x_test.device).normal_(0, 1)

if args.train_from == "":
    decoder = Decoder()
    nu = Nu()
    encoder = Encoder()
    if gpu:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        nu = nu.cuda()
    nu_optimizer = optim.Adam(nu.parameters(), lr=args.nu_lr)
    end2end_optimizer = optim.Adam(chain(encoder.parameters(), decoder.parameters()), lr=args.end2end_lr)
else:
    logging.info('load model from' + args.train_from)
    checkpoint = torch.load(args.train_from, map_location="cuda:" + str(args.gpu) if gpu else 'cpu')
    args = checkpoint['args']
    logging.info(str(args))
    nu = checkpoint['nu']
    nu_optimizer = checkpoint['nu_optimizer']
    encoder = checkpoint['encoder']
    decoder = checkpoint['decoder']
    end2end_optimizer = checkpoint['end2end_optimizer']


def evaluation(x, eps, epo=None):
    z_x = encoder(x, eps)

    z_plt = z_x.detach().cpu().numpy()
    if epo is not None:
        create_scatter(z_plt, save_path=os.path.join(args.results_folder, '%06d.png' % epo))
    else:
        create_scatter(z_plt, save_path=os.path.join(args.results_folder, 'evaluation.png'))

    pi = decoder(z_x)
    rec = -torch.mean(torch.log(pi[x == 1])) - torch.sum(torch.log(1 - pi[x == 0])) / x.shape[0]
    logging.info('rec: %.4f' % rec)
    z = eps.clone()
    kl = torch.mean(nu(x, z_x)) - torch.mean(torch.exp(nu(x, z))) + 1.0
    logging.info('kl with nu: %.4f' % kl)
    logging.info('neg_ELBO with nu: %.4f' % (rec + kl))


def check_point(epo=None):
    check_pt = {
        'args': args,
        'nu': nu,
        'nu_optimizer': nu_optimizer,
        'encoder': encoder,
        'decoder': decoder,
        'end2end_optimizer': end2end_optimizer
    }
    if epo is not None:
        torch.save(check_pt, os.path.join(args.results_folder, '%06d.pt' % epo))
    else:
        torch.save(check_pt, os.path.join(args.results_folder, 'checkpoint.pt'))


if args.test:
    evaluation(x_test, eps_test)
    exit()

pbar = tqdm(range(args.num_epochs + 1))
for epo in pbar:

    if epo % args.plot_freq == 0:
        logging.info('------------------------------------------------------')
        logging.info("the current epo is %d" % epo)
        evaluation(x_test, eps_test, epo=epo)
        check_point(epo)

    x = torch.zeros((4 * args.num_particles, 4), device='cuda' if gpu else 'cpu')
    rand_N = torch.LongTensor(4).random_(0, 4).tolist()
    rand = [j for i in rand_N for j in [i] * args.num_particles]
    idx = torch.LongTensor(rand).to(x.device)
    x.scatter_(1, idx.view(-1, 1), 1.0)
    eps = torch.zeros((4 * args.num_particles, 2), device=x.device).normal_(0, 1)
    z_x = encoder(x, eps)

    # nu update
    for k in torch.arange(args.num_nu_updates):
        z_x_nu = z_x.detach()
        z = torch.zeros((4 * args.num_particles, 2), device=x.device).normal_(0, 1)
        nu_loss = torch.mean(torch.exp(nu(x, z)) - nu(x, z_x_nu))
        optimizer_step(nu_loss, nu_optimizer)

    # end2end update
    pi = decoder(z_x)
    rec = - torch.mean(torch.log(pi[x == 1])) - torch.sum(torch.log(1 - pi[x == 0])) / x.shape[0]
    loss = rec + torch.mean(nu(x, z_x))
    optimizer_step(loss, end2end_optimizer)

    torch.cuda.empty_cache()
