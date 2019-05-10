
import sys
import os
from preprocess_ptb import Indexer
import torch
import argparse
import logging
from data import Dataset
import numpy as np

parser = argparse.ArgumentParser()

# global parameters
parser.add_argument('--test_file', default='data/ptb-test.hdf5')
parser.add_argument('--results_folder_prefix', default='results_')
parser.add_argument('--train_from_epo', default=40, type=int)
parser.add_argument('--seed', default=63, type=int)
parser.add_argument('--log_prefix', default='interpolation')
parser.add_argument('--model', default='mle', type=str, choices=['mle', 'mle_mi'])
parser.add_argument('--num_particles_eval', default=24, type=int)
parser.add_argument('--latent_dim', default=32, type=int)

# use GPU
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--no_gpu', action="store_true")

if sys.argv[1:] == ['0', '0']:
    args = parser.parse_args([])   # run in pycharm console
else:
    args = parser.parse_args()  # run in cmd

# parameters
test_data = Dataset(args.test_file)
test_sents = test_data.batch_size.sum()
vocab_size = int(test_data.vocab_size)
vocab = Indexer()
vocab.load_vocab('data/ptb.dict')

print('Test data: %d batches' % len(test_data))
print('Test data: %d sentences' % test_sents)
print('Word vocab size: %d' % vocab_size)

results_folder = args.results_folder_prefix + args.model + '/'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
logging.basicConfig(filename=os.path.join(results_folder, args.log_prefix + '.log'),
                    level=logging.INFO, format='%(asctime)s--- %(message)s')
if not torch.cuda.is_available(): args.no_gpu = True
gpu = not args.no_gpu
if gpu: torch.cuda.set_device(args.gpu)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if gpu: torch.cuda.manual_seed(args.seed)

train_from = results_folder + '%03d.pt' % args.train_from_epo

logging.info('load model from' + train_from)
checkpoint = torch.load(train_from, map_location="cuda:" + str(args.gpu) if gpu else 'cpu')

encoder = checkpoint['encoder']
decoder = checkpoint['decoder']

if gpu:
    encoder = encoder.cuda()
    decoder = decoder.cuda()


def get_text(sents):
    sampled_sents = []
    for i in range(sents.shape[0]):
        decoded_sentence = [vocab.idx2word[int(s)] for s in sents[i, :]]
        sampled_sents.append(decoded_sentence)
    for i, sent in enumerate(sampled_sents):
        logging.info(('the %d-th sent: ') % i + ' '.join(sent))


def get_avg_code(encoder, sents):
    if gpu:
        sents = sents.cuda()
    eps = torch.randn((sents.shape[0], args.latent_dim), device=sents.device)
    z_x, _ = encoder(sents, eps)
    z_x = z_x.mean(dim=0, keepdim=True).data
    return z_x


def sample_text(decoder, input, z, EOS):
    sentence = [input.item()]
    max_index = 0

    input_word = input
    batch_size, n_sample, _ = z.size()
    seq_len = 1
    z_ = z.expand(batch_size, seq_len, args.latent_dim)

    word_vecs = decoder.dec_word_vecs(input_word)
    decoder.h0 = torch.zeros((decoder.dec_num_layers, word_vecs.size(0), decoder.dec_h_dim), device=z.device)
    decoder.c0 = torch.zeros((decoder.dec_num_layers, word_vecs.size(0), decoder.dec_h_dim), device=z.device)
    decoder.h0[-1] = decoder.latent_hidden_linear(z)
    hidden = None

    while max_index != EOS and len(sentence) < 100:
        # (batch_size, seq_len, ni)
        word_embed = decoder.dec_word_vecs(input_word)
        word_embed = torch.cat((word_embed, z_), -1)

        if len(sentence) == 1:
            output, hidden = decoder.dec_rnn(word_embed, (decoder.h0, decoder.c0))
        else:
            output, hidden = decoder.dec_rnn(word_embed, hidden)

        preds = decoder.dec_linear[1:](output.view(word_vecs.size(0) * word_vecs.size(1), -1)).view(-1)
        max_index = torch.argmax(preds).item()
        input_word = torch.ones((), dtype=torch.long).new_tensor([[max_index]])
        if gpu: input_word = input_word.cuda()
        sentence.append(max_index)

    return sentence


sents, length, batch_size = test_data[12]  # sample from the 16th batch
batch_size = batch_size.item()
length = length.item()
if gpu: sents = sents.cuda()
logging.info('---------------- Original sentences: ----------------')
random_ind = [3, 21]
get_text(sents[random_ind, :])

# sample 2 sents for interpolation
encoder.eval()
decoder.dec_linear.eval()
decoder.dropout.eval()
sents1 = sents[[random_ind[0]] * args.num_particles_eval, :]
sents2 = sents[[random_ind[1]] * args.num_particles_eval, :]

z_x1 = get_avg_code(encoder, sents1)
z_x2 = get_avg_code(encoder, sents2)
z_x = z_x1
for t in torch.arange(0.1, 1.01, 0.1):
    z_x = torch.cat((z_x, z_x1 * (1 - t) + z_x2 * t), dim=0)

logging.info('---------------- Sample sentences: ----------------')
sampled_sents = []

for i in range(11):
    z = z_x[i, :]
    z = z.view(1, 1, -1)

    start = vocab.convert('<s>')
    START = torch.ones((), dtype=torch.long).new_tensor([[start]])
    end = vocab.convert('</s>')
    if gpu: START = START.cuda()
    sentence = sample_text(decoder, START, z, end)
    decoded_sentence = [vocab.idx2word[s] for s in sentence]
    sampled_sents.append(decoded_sentence)

for i, sent in enumerate(sampled_sents):
    logging.info(('the %d-th interpolated sent: ') % i + ' '.join(sent))
