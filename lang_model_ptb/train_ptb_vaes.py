
import sys
import os
import torch
from torch import optim
import argparse
import logging
import torch.nn as nn
from torch.autograd import Variable
from data import Dataset
import numpy as np
import math
from optim_n2n import OptimN2N
from models_ptb import RNNVAE
import utils_ptb as utils
from preprocess_ptb import Indexer
from tqdm import tqdm

parser = argparse.ArgumentParser()

# global parameters
parser.add_argument('--train_file', default='data/ptb-train.hdf5')
parser.add_argument('--val_file', default='data/ptb-val.hdf5')
parser.add_argument('--test_file', default='data/ptb-test.hdf5')
parser.add_argument('--results_folder_prefix', default='results_')
parser.add_argument('--train_from', default='')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--test', action="store_true")
parser.add_argument('--log_prefix', default='eval')
parser.add_argument('--model', default='vae', type=str, choices=['vae', 'beta_vae', 'savae', 'cyc_vae'])

# KL cost annealing, increase beta from beta_0 by 1/warmup in certain steps
parser.add_argument('--warmup', default=10, type=int)
parser.add_argument('--beta_0', default=0.1, type=float)
parser.add_argument('--beta', default=0.5, type=float)

# global training parameters
parser.add_argument('--num_epochs', default=40, type=int)

# use GPU
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--no_gpu', action="store_true")

# model and optimizer parameters
parser.add_argument('--latent_dim', default=32, type=int)
parser.add_argument('--enc_word_dim', default=256, type=int)
parser.add_argument('--enc_h_dim', default=256, type=int)
parser.add_argument('--enc_num_layers', default=1, type=int)

parser.add_argument('--dec_word_dim', default=256, type=int)
parser.add_argument('--dec_h_dim', default=256, type=int)
parser.add_argument('--dec_num_layers', default=1, type=int)
parser.add_argument('--dec_dropout', default=0.5, type=float)

parser.add_argument('--max_grad_norm', default=5.0, type=float)
parser.add_argument('--lr', default=1e-3, type=float)

# savae parameters
parser.add_argument('--svi_steps', default=10, type=int)
parser.add_argument('--svi_lr1', default=1, type=float)
parser.add_argument('--svi_lr2', default=1, type=float)
parser.add_argument('--eps', default=1e-5, type=float)
parser.add_argument('--decay', default=0, type=int)
parser.add_argument('--momentum', default=0.5, type=float)
parser.add_argument('--svi_max_grad_norm', default=5, type=float)

# cyc_vae parameters
parser.add_argument('--cycle', type=int, default=10)

if sys.argv[1:] == ['0', '0']:
    args = parser.parse_args([])   # run in pycharm console
else:
    args = parser.parse_args()  # run in cmd

# parameters
train_data = Dataset(args.train_file)
val_data = Dataset(args.val_file)
test_data = Dataset(args.test_file)
train_sents = train_data.batch_size.sum()
val_sents = val_data.batch_size.sum()
test_sents = test_data.batch_size.sum()
vocab_size = int(train_data.vocab_size)
vocab = Indexer()
vocab.load_vocab('data/ptb.dict')

print('Train data: %d batches' % len(train_data))
print('Val data: %d batches' % len(val_data))
print('Test data: %d batches' % len(test_data))
print('Train data: %d sentences' % train_sents)
print('Val data: %d sentences' % val_sents)
print('Test data: %d sentences' % test_sents)
print('Word vocab size: %d' % vocab_size)

results_folder = args.results_folder_prefix + args.model + '/'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
logging.basicConfig(filename=os.path.join(results_folder, args.log_prefix + '.log'),
                    level=logging.INFO, format='%(asctime)s--- %(message)s')
logging.info("the configuration:")
logging.info(str(args).replace(',', '\n'))
if not torch.cuda.is_available(): args.no_gpu = True
gpu = not args.no_gpu
if gpu: torch.cuda.set_device(args.gpu)

np.random.seed(args.seed)
prng = np.random.RandomState()
torch.manual_seed(args.seed)
if gpu: torch.cuda.manual_seed(args.seed)


beta = args.beta_0
epo_0 = 0

model = RNNVAE(vocab_size=vocab_size,
               enc_word_dim=args.enc_word_dim,
               enc_h_dim=args.enc_h_dim,
               enc_num_layers=args.enc_num_layers,
               dec_word_dim=args.dec_word_dim,
               dec_h_dim=args.dec_h_dim,
               dec_num_layers=args.dec_num_layers,
               dec_dropout=args.dec_dropout,
               latent_dim=args.latent_dim,
               mode=args.model)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

if args.train_from == "":
    for param in model.parameters():
        param.data.uniform_(-0.1, 0.1)
    if gpu:
        model = model.cuda()
        criterion.cuda()
else:
    logging.info('load model from' + args.train_from)
    checkpoint = torch.load(args.train_from, map_location="cuda:" + str(args.gpu) if gpu else 'cpu')

    if not args.test:  # if testing, start from random seed above; if continuing training, have a 'real' restart
        np_random_state = checkpoint['np_random_state']
        prng.set_state(np_random_state)
        torch_rng_state = checkpoint['torch_rng_state']
        torch_rng_state_cuda = checkpoint['torch_rng_state_cuda']
        torch.set_rng_state(torch_rng_state.cpu())
        if gpu: torch.cuda.set_rng_state(torch_rng_state_cuda.cpu())

    model = checkpoint['model']
    criterion = checkpoint['criterion']
    optimizer = checkpoint['optimizer']
    beta = checkpoint['beta']
    epo_0 = int(args.train_from[-6:-3])

logging.info("model configuration:")
logging.info(str(model))


def variational_loss(input, sents, model, beta, z=None):
    mean, logvar = input
    z_samples = model._reparameterize(mean, logvar, z)
    preds = model._dec_forward(sents, z_samples)
    nll = sum([criterion(preds[:, l], sents[:, l + 1]) for l in range(preds.size(1))])
    kl = utils.kl_loss_diag(mean, logvar)
    return nll + beta * kl


update_params = list(model.dec.parameters())
meta_optimizer = OptimN2N(variational_loss, model, update_params, beta, eps=args.eps,
                          lr=[args.svi_lr1, args.svi_lr2],
                          iters=args.svi_steps, momentum=args.momentum,
                          acc_param_grads=True,
                          max_grad_norm=args.svi_max_grad_norm)


def evaluation(data, model, meta_optimizer):
    model.dec_linear.eval()
    model.dropout.eval()

    meta_optimizer.beta = 1.0

    num_sents = 0.0
    num_words = 0.0
    total_rec = 0.0
    total_kl = 0.0
    total_mi = 0.0
    total_mean_au = torch.zeros(args.latent_dim, device='cuda' if gpu else 'cpu')
    total_sq_au = torch.zeros(args.latent_dim, device='cuda' if gpu else 'cpu')

    pbar = tqdm(range(len(data)))
    for i in pbar:
        # logging.info('batch: %d' % i)

        sents, length, batch_size = data[i]
        batch_size = batch_size.item()
        length = length.item()
        num_sents += batch_size
        num_words += batch_size * length
        if gpu: sents = sents.cuda()

        mean, logvar = model._enc_forward(sents)
        if args.model == 'savae':
            mean_svi = Variable(mean.data, requires_grad=True)
            logvar_svi = Variable(logvar.data, requires_grad=True)
            var_params_svi = meta_optimizer.forward([mean_svi, logvar_svi], sents)
            mean, logvar = var_params_svi

        # rec, kl
        mean, logvar = mean.data, logvar.data
        z_samples = model._reparameterize(mean, logvar)
        preds = model._dec_forward(sents, z_samples)
        nll_vae = sum([criterion(preds[:, l], sents[:, l + 1]) for l in range(length)])
        total_rec += nll_vae.item() * batch_size
        kl_vae = utils.kl_loss_diag(mean, logvar)
        total_kl += kl_vae.item() * batch_size

        # active units
        total_mean_au += torch.sum(mean, dim=0)
        total_sq_au += torch.sum(mean ** 2, dim=0)

        # mutual information
        x_batch, nz = mean.size()
        # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+logvar).sum(-1)
        neg_entropy = (-0.5 * nz * math.log(2 * math.pi) - 0.5 * (1 + logvar).sum(-1)).mean()

        # [z_batch, 1, nz]
        z_samples = z_samples.unsqueeze(1)

        # [1, x_batch, nz]
        mean, logvar = mean.unsqueeze(0), logvar.unsqueeze(0)
        var = logvar.exp()

        # (z_batch, x_batch, nz)
        dev = z_samples - mean

        # (z_batch, x_batch)
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - 0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

        # log q(z): aggregate posterior
        # [z_batch]
        log_qz = utils.log_sum_exp(log_density, dim=1) - math.log(x_batch)

        total_mi += (neg_entropy - log_qz.mean(-1)).item() * batch_size

        del mean, logvar
        torch.cuda.empty_cache()

    rec = total_rec / num_sents
    kl = total_kl / num_sents
    nelbo = rec + kl
    ppl = math.exp((total_rec + total_kl) / num_words)
    mi = total_mi / num_sents

    logging.info('rec: %.4f' % rec)
    logging.info('kl with nu: %.4f' % kl)
    logging.info('neg_ELBO with nu: %.4f' % nelbo)
    logging.info('ppl: %.4f' % ppl)
    logging.info('mi: %.4f' % mi)

    mean_au = total_mean_au / num_sents
    sq_au = total_sq_au / num_sents
    au_cov = sq_au - mean_au ** 2
    au = (au_cov >= 0.01).sum().item()
    logging.info('au_cov: %s' % str(au_cov))
    logging.info('au: %.4f' % au)

    report = "rec %f, kl %f, elbo %f, \nppl %f, mi %f, au %f\n" % (rec, kl, nelbo, ppl, mi, au)
    print(report)

    model.train()

    return nelbo


def check_point(epo=None):
    check_pt = {
        'args': args,
        'model': model,
        'criterion': criterion,
        'optimizer': optimizer,
        'beta': beta,

        'np_random_state': prng.get_state(),
        'torch_rng_state': torch.get_rng_state(),
        'torch_rng_state_cuda': torch.cuda.get_rng_state() if gpu else torch.get_rng_state()
    }
    if epo is not None:
        torch.save(check_pt, os.path.join(results_folder, '%03d.pt' % epo))
    else:
        torch.save(check_pt, os.path.join(results_folder, 'checkpoint.pt'))


if args.test:
    logging.info('\n------------------------------------------------------')
    logging.info("evaluation:")
    evaluation(test_data, model, meta_optimizer)
    exit()


logging.info('\n------------------------------------------------------')
logging.info("the current epo is %d of %d" % (epo_0, args.num_epochs))
print("the current epo is %d of %d" % (epo_0, args.num_epochs))
logging.info("evaluation:")
print("evaluation:")
check_point(epo_0)
evaluation(test_data, model, meta_optimizer)

for epo in torch.arange(epo_0 + 1, args.num_epochs + 1):
    logging.info('\n------------------------------------------------------')
    logging.info("the current epo is %d of %d" % (epo, args.num_epochs))
    print("the current epo is %d of %d" % (epo, args.num_epochs))
    logging.info("training:")
    print("training:")

    # training
    model.train()

    if args.model == 'cyc_vae':
        if (epo - 1) % args.cycle == 0:
            beta = 0.1
            logging.info('KL annealing restart')

    random_bat = torch.randperm(len(train_data)).tolist()
    pbar = tqdm(range(len(train_data)))
    for bat in pbar:
        mini_batch = random_bat[bat]
        sents, length, batch_size = train_data[mini_batch]
        batch_size = batch_size.item()
        length = length.item()
        if gpu: sents = sents.cuda()

        if args.model == 'vae' or args.model == 'cyc_vae' or args.model == 'savae':
            if args.warmup > 0:
                beta = min(1.0, beta + 1. / (args.warmup * len(train_data)))
        if args.model == 'beta_vae':
            if args.warmup > 0:
                beta = min(args.beta, beta + 1. / (args.warmup * len(train_data)))
        meta_optimizer.beta = beta

        optimizer.zero_grad()

        mean, logvar = model._enc_forward(sents)
        if args.model == 'vae' or args.model == 'beta_vae' or args.model == 'cyc_vae':
            z_samples = model._reparameterize(mean, logvar)
            preds = model._dec_forward(sents, z_samples)
            nll_vae = sum([criterion(preds[:, l], sents[:, l + 1]) for l in range(length)])
            kl_vae = utils.kl_loss_diag(mean, logvar)
            vae_loss = nll_vae + beta * kl_vae
            vae_loss.backward()
            del vae_loss
        if args.model == 'savae':
            var_params = torch.cat([mean, logvar], 1)
            mean_svi = Variable(mean.data, requires_grad=True)
            logvar_svi = Variable(logvar.data, requires_grad=True)
            var_params_svi = meta_optimizer.forward([mean_svi, logvar_svi], sents, False)
            mean_svi_final, logvar_svi_final = var_params_svi
            z_samples = model._reparameterize(mean_svi_final, logvar_svi_final)
            preds = model._dec_forward(sents, z_samples)
            nll_svi = sum([criterion(preds[:, l], sents[:, l + 1]) for l in range(length)])
            kl_svi = utils.kl_loss_diag(mean_svi_final, logvar_svi_final)
            var_loss = nll_svi + beta * kl_svi
            var_loss.backward()

            var_param_grads = meta_optimizer.backward([mean_svi_final.grad, logvar_svi_final.grad], False)
            var_param_grads = torch.cat(var_param_grads, 1)
            var_params.backward(var_param_grads)
            del var_loss, var_param_grads
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        torch.cuda.empty_cache()
        assert not torch.isnan(mean).any(), 'training get nan z mean'

    # evaluation
    logging.info("evaluation:")
    print("evaluation:")
    check_point(epo)
    if epo > 30: evaluation(test_data, model, meta_optimizer)
