
import sys
import os
from preprocess_ptb import Indexer
import torch
import argparse
import logging
from data import Dataset
import numpy as np
import math
from tqdm import tqdm
import kenlm

parser = argparse.ArgumentParser()

# global parameters
parser.add_argument('--train_file', default='data/train.txt')
parser.add_argument('--results_folder_prefix', default='results_')
parser.add_argument('--train_from_epo', default=40, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--log_prefix', default='generative_model')
parser.add_argument('--generate_text', default='generated.txt')
parser.add_argument('--model', default='ae', type=str, choices=['mle','mle_mi','ae','vae','beta_vae','savae','cyc_vae'])
parser.add_argument('--num_particles_eval', default=128, type=int)
parser.add_argument('--latent_dim', default=32, type=int)

# use GPU
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--no_gpu', action="store_true")

if sys.argv[1:] == ['0', '0']:
    args = parser.parse_args([])   # run in pycharm console
else:
    args = parser.parse_args()  # run in cmd

# parameters
vocab = Indexer()
vocab.load_vocab('data/ptb.dict')
vocab_size = len(vocab.idx2word)
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

if args.model == 'mle' or args.model == 'mle_mi' or args.model == 'ae':
    decoder = checkpoint['decoder']
else:
    decoder = checkpoint['model']

if gpu:
    decoder = decoder.cuda()
decoder.dec_linear.eval()
decoder.dropout.eval()


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

    while max_index != EOS and len(sentence) < 80:
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


def train_ngram_lm(kenlm_path, data_path, output_path, N):
    """
    Trains a modified Kneser-Ney n-gram KenLM from a text file.
    Creates a .arpa file to store n-grams.
    """
    # create .arpa file of n-grams
    curdir = os.path.abspath(os.path.curdir)

    command = "bin/lmplz -o " + str(N) + ' --skip_symbols "<unk>" --discount_fallback' \
              + " <" + os.path.join(curdir, data_path) + " >" + os.path.join(curdir, output_path)
    os.system("cd " + os.path.join(kenlm_path, 'build') + " && " + command)

    # create language model
    model = kenlm.Model(output_path)

    return model


def get_ppl(lm, sentences):
    """
    Assume sentences is a list of strings (space delimited sentences)
    """
    total_nll = 0
    total_wc = 0
    for sent in sentences:
        words = sent.strip().split()
        score = lm.score(sent, bos=True, eos=False)
        word_count = len(words)
        total_wc += word_count
        total_nll += score
    ppl = 10 ** -(total_nll / total_wc)
    return ppl


# generate sentences from decoder
sampled_sents = []
pbar = tqdm(range(10))
for times in pbar:
    z_prior = torch.randn((args.num_particles_eval, args.latent_dim), device='cuda' if gpu else 'cpu')

    for i in range(args.num_particles_eval):
        z = z_prior[i, :]
        z = z.view(1, 1, -1)

        start = vocab.convert('<s>')
        START = torch.ones((), dtype=torch.long).new_tensor([[start]])
        end = vocab.convert('</s>')
        if gpu: START = START.cuda()
        sentence = sample_text(decoder, START, z, end)
        decoded_sentence = [vocab.idx2word[s] for s in sentence]
        sampled_sents.append(' '.join(decoded_sentence[1:-1]))

output_file = "{}{}".format(results_folder, args.generate_text)
with open(output_file, 'w') as f:
    # laplacian smoothing
    for word in vocab.idx2word.values():
        f.write(word + "\n")
    for i in range(len(sampled_sents)):
        # real sentence
        f.write(sampled_sents[i])
        if i != len(sampled_sents) - 1:
            f.write("\n")

# forward and reverse PPL
with open(args.train_file, 'r') as f:
    lines = f.readlines()
sentences = [l.replace('\n', '') for l in lines]

kenlm_model_train = train_ngram_lm('./kenlm/', args.train_file, '{}lm_train.arpa'.format(results_folder), 5)
forward = get_ppl(kenlm_model_train, sampled_sents)

kenlm_model_generated = train_ngram_lm('./kenlm/', output_file, '{}lm_generated.arpa'.format(results_folder), 5)
reverse = get_ppl(kenlm_model_generated, sentences)

print('forward {:4.3f} | reverse {:4.3f}'.format(forward, reverse))
logging.info('forward {:4.3f} | reverse {:4.3f}'.format(forward, reverse))
