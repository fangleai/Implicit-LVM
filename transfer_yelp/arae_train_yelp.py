
import argparse
import os
import time
import math
import numpy as np
import random
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from utils_yelp import to_gpu, Corpus, batchify, train_ngram_lm, get_ppl
from arae_models_yelp import Seq2Seq2Decoder, Seq2Seq, MLP_D, MLP_G, MLP_Classify
import logging
from tqdm import tqdm
import fasttext
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

parser = argparse.ArgumentParser(description='Yelp style transfer')
# Path Arguments
parser.add_argument('--data_path', type=str, default='./data/', help='location of the data corpus')
parser.add_argument('--log_prefix', default='')
parser.add_argument('--load_path', type=str, default='', help='location to load pre-trained models')  # ./output/WAE/
parser.add_argument('--load_epoch', type=int, default=0, help='load epoch #')
parser.add_argument('--model', type=str, default='ARAE', help='model name')
parser.add_argument('--load_vocab', type=str, default="", help='path to load vocabulary from')
parser.add_argument('--eval', action='store_true', help='evaluation')
parser.add_argument('--kenlm_path', type=str, default='./kenlm/', help='kenlm path')
parser.add_argument('--N', type=int, default=5, help='kenlm N-gram LM')

# Data Processing Arguments
parser.add_argument('--vocab_size', type=int, default=30000, help='cut vocab to #. most frequently seen words in train')
parser.add_argument('--maxlen', type=int, default=25, help='maximum sentence length')
parser.add_argument('--lowercase', dest='lowercase', action='store_true', help='lowercase all text')
parser.add_argument('--no-lowercase', dest='lowercase', action='store_true', help='not lowercase all text')
parser.set_defaults(lowercase=True)

# Model Arguments
parser.add_argument('--emsize', type=int, default=128, help='size of word embeddings')
parser.add_argument('--nhidden', type=int, default=128, help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1, help='number of layers')
parser.add_argument('--noise_r', type=float, default=0.1, help='stdev of noise for autoencoder (regularizer)')
parser.add_argument('--noise_anneal', type=float, default=0.9995,
                    help='anneal noise_r exponentially by % every 100 iters')
parser.add_argument('--hidden_init', action='store_true', help="initialize decoder hidden state with encoder's")
parser.add_argument('--arch_g', type=str, default='128-128', help='generator architecture (MLP)')
parser.add_argument('--arch_d', type=str, default='128-128', help='critic/discriminator architecture (MLP)')
parser.add_argument('--arch_classify', type=str, default='128-128', help='classifier architecture')
parser.add_argument('--z_size', type=int, default=32, help='dimension of random noise z to feed into generator')
parser.add_argument('--temp', type=float, default=1, help='softmax temperature (lower --> more discrete)')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout applied to layers (0 = no dropout)')

# Training Arguments
parser.add_argument('--epochs', type=int, default=25, help='maximum number of epochs')
parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='batch size')
parser.add_argument('--niters_gan_d', type=int, default=4, help='number of discriminator iterations in training')
parser.add_argument('--lr_ae', type=float, default=1, help='autoencoder learning rate')
parser.add_argument('--lr_gan_g', type=float, default=1e-04, help='generator learning rate')
parser.add_argument('--lr_gan_d', type=float, default=1e-04, help='critic/discriminator learning rate')
parser.add_argument('--lr_classify', type=float, default=1e-04, help='classifier learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--clip', type=float, default=1, help='gradient clipping, max norm')
parser.add_argument('--gan_gp_lambda', type=float, default=0.1, help='WGAN GP penalty lambda')
parser.add_argument('--grad_lambda', type=float, default=0.01, help='WGAN into AE lambda')
parser.add_argument('--lambda_class', type=float, default=1, help='lambda on classifier')

# Evaluation Arguments
parser.add_argument('--sample', action='store_true', help='sample when decoding for generation')

# Other
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--no_gpu', action="store_true")

if sys.argv[1:] == ['0', '0']:
    args = parser.parse_args([])  # run in pycharm console
else:
    args = parser.parse_args()  # run in cmd

# Set the random seed manually for reproducibility.
random.seed(args.seed)
np.random.seed(args.seed)
prng = np.random.RandomState()
torch.manual_seed(args.seed)
if not torch.cuda.is_available(): args.no_gpu = True
use_cuda = not args.no_gpu
if use_cuda:
    torch.cuda.set_device(args.gpu_id)  # set gpu device
    torch.cuda.manual_seed(args.seed)

# make output directory if it doesn't already exist
if not os.path.isdir('./output'):
    os.makedirs('./output')
if not os.path.isdir('./output/{}'.format(args.model)):
    os.makedirs('./output/{}'.format(args.model))
args.outf = './output/{}'.format(args.model)
###############################################################################
# Load data
###############################################################################
# (Path to textfile, Name, Use4Vocab)
datafiles = [(os.path.join(args.data_path, "test0.txt"), "test0", False),
             (os.path.join(args.data_path, "test1.txt"), "test1", False),
             (os.path.join(args.data_path, "valid0.txt"), "valid0", False),
             (os.path.join(args.data_path, "valid1.txt"), "valid1", False),
             (os.path.join(args.data_path, "train0.txt"), "train0", True),
             (os.path.join(args.data_path, "train1.txt"), "train1", True)]
vocabdict = None
if args.load_vocab != "":
    vocabdict = json.load(args.vocab)
    vocabdict = {k: int(v) for k, v in vocabdict.items()}
corpus = Corpus(datafiles,
                maxlen=args.maxlen,
                vocab_size=args.vocab_size,
                lowercase=args.lowercase,
                vocab=vocabdict)

# save arguments
ntokens = len(corpus.dictionary.word2idx)
print("Vocabulary Size: {}".format(ntokens))
args.ntokens = ntokens
logging.basicConfig(filename=os.path.join('./output/', args.model, args.log_prefix + 'logs.txt'),
                    level=logging.INFO, format='%(asctime)s--- %(message)s')
logging.info("the configuration:")
logging.info(str(args).replace(',', '\n'))

# dumping vocabulary
with open('{}/vocab.json'.format(args.outf), 'w') as f:
    json.dump(corpus.dictionary.word2idx, f)
with open('{}/args.json'.format(args.outf), 'w') as f:
    json.dump(vars(args), f)

eval_batch_size = 100
test0_data = batchify(corpus.data['test0'], eval_batch_size, shuffle=False)
test1_data = batchify(corpus.data['test1'], eval_batch_size, shuffle=False)
train0_data = batchify(corpus.data['train0'], args.batch_size, shuffle=True)
train1_data = batchify(corpus.data['train1'], args.batch_size, shuffle=True)

print("Loaded data!")

fixed_noise = to_gpu(use_cuda, Variable(torch.ones(args.batch_size, args.z_size)))
fixed_noise.data.normal_(0, 1)
one = to_gpu(use_cuda, torch.FloatTensor([1]))
mone = one * -1

# fasttext library
fasttext_classifier = fasttext.supervised('./data/data.txt', 'model', label_prefix='__label__')
result = fasttext_classifier.test('./data/data.txt')
print('P@1:', result.precision)
print('R@1:', result.recall)
print('Number of examples:', result.nexamples)

###############################################################################
# Build the models
###############################################################################

ntokens = len(corpus.dictionary.word2idx)
autoencoder = Seq2Seq2Decoder(emsize=args.emsize,
                              nhidden=args.nhidden,
                              ntokens=ntokens,
                              nlayers=args.nlayers,
                              noise_r=args.noise_r,
                              hidden_init=args.hidden_init,
                              dropout=args.dropout,
                              gpu=use_cuda)

gan_gen = MLP_G(ninput=args.z_size, noutput=args.nhidden, layers=args.arch_g)
gan_disc = MLP_D(ninput=args.nhidden, noutput=1, layers=args.arch_d)
classifier = MLP_Classify(ninput=args.nhidden, noutput=1, layers=args.arch_classify)

logging.info("model configuration:")
logging.info(str(autoencoder))
logging.info(str(gan_gen))
logging.info(str(gan_disc))
logging.info(str(classifier))

optimizer_ae = optim.SGD(autoencoder.parameters(), lr=args.lr_ae)
optimizer_gan_g = optim.Adam(gan_gen.parameters(), lr=args.lr_gan_g, betas=(args.beta1, 0.999))
optimizer_gan_d = optim.Adam(gan_disc.parameters(), lr=args.lr_gan_d, betas=(args.beta1, 0.999))
optimizer_classify = optim.Adam(classifier.parameters(), lr=args.lr_classify, betas=(args.beta1, 0.999))
criterion_ce = nn.CrossEntropyLoss()

if use_cuda:
    autoencoder = autoencoder.cuda()
    gan_gen = gan_gen.cuda()
    gan_disc = gan_disc.cuda()
    classifier = classifier.cuda()
    criterion_ce = criterion_ce.cuda()


def evaluate_generator(whichdecoder, noise, epoch):
    gan_gen.eval()
    autoencoder.eval()

    # generate from fixed random noise
    fake_hidden = gan_gen(noise)
    max_indices = autoencoder.generate(whichdecoder, fake_hidden, maxlen=args.maxlen, sample=args.sample)

    with open("%s/%s_generated%d.txt" % (args.outf, epoch, whichdecoder), "w") as f:
        max_indices = max_indices.data.cpu().numpy()
        for idx in max_indices:
            # generated sentence
            words = [corpus.dictionary.idx2word[x] for x in idx]
            # truncate sentences to first occurrence of <eos>
            truncated_sent = []
            for w in words:
                if w != '<eos>':
                    truncated_sent.append(w)
                else:
                    break
            chars = " ".join(truncated_sent)
            f.write(chars)
            f.write("\n")


def evaluate_autoencoder(whichdecoder, data_source, epoch):
    # Turn on evaluation mode which disables dropout.
    eos_id = corpus.dictionary.word2idx['<eos>']
    autoencoder.eval()
    ntokens = len(corpus.dictionary.word2idx)
    n_sents = 0.0
    total_loss = 0.0
    token_accuracies = 0.0
    all_source_sents = []
    all_transfer_sents = []

    pbar = tqdm(range(len(data_source)))
    for ii in pbar:
        batch = data_source[ii]

        source, target, lengths = batch
        source = to_gpu(use_cuda, Variable(source, requires_grad=False))
        target = to_gpu(use_cuda, Variable(target, requires_grad=False))
        n_sents += source.size()[0]

        mask = target.gt(0)
        masked_target = target.masked_select(mask)
        # examples x ntokens
        output_mask = mask.unsqueeze(1).expand(mask.size(0), ntokens)

        hidden = autoencoder(0, source, lengths, noise=False, encode_only=True)

        # output: batch x seq_len x ntokens
        if whichdecoder == 0:
            output = autoencoder(0, source, lengths, noise=False)
            flattened_output = output.view(-1, ntokens)
            masked_output = flattened_output.masked_select(output_mask).view(-1, ntokens)
            # accuracy
            max_vals1, max_indices1 = torch.max(masked_output, 1)
            token_accuracies += torch.mean(max_indices1.eq(masked_target).float()).item()

            max_values1, max_indices1 = torch.max(output, 2)
            max_indices2 = autoencoder.generate(1, hidden, maxlen=50)
        else:
            output = autoencoder(1, source, lengths, noise=False)
            flattened_output = output.view(-1, ntokens)
            masked_output = flattened_output.masked_select(output_mask).view(-1, ntokens)
            # accuracy
            max_vals2, max_indices2 = torch.max(masked_output, 1)
            token_accuracies += torch.mean(max_indices2.eq(masked_target).float()).item()

            max_values2, max_indices2 = torch.max(output, 2)
            max_indices1 = autoencoder.generate(0, hidden, maxlen=50)

        # forward
        total_loss += criterion_ce(masked_output / args.temp, masked_target).data

        # all_source_sents, all_transfer_sents
        max_indices1 = max_indices1.view(output.size(0), -1).data.cpu().numpy()
        max_indices2 = max_indices2.view(output.size(0), -1).data.cpu().numpy()
        target = target.view(output.size(0), -1).data.cpu().numpy()
        tran_indices = max_indices2 if whichdecoder == 0 else max_indices1
        for t, tran_idx in zip(target, tran_indices):
            # real sentence
            truncated_to_eos = t.tolist().index(eos_id) if eos_id in t.tolist() else len(t)
            chars = " ".join([corpus.dictionary.idx2word[x] for x in t[:truncated_to_eos]])
            all_source_sents.append(chars)
            # transfer sentence
            truncated_to_eos = tran_idx.tolist().index(eos_id) if eos_id in tran_idx.tolist() else len(tran_idx)
            chars = " ".join([corpus.dictionary.idx2word[x] for x in tran_idx[:truncated_to_eos]])
            all_transfer_sents.append(chars)

    # compare the original and transfer
    aeoutf_from = "{}/{}_output_decoder_{}_from.txt".format(args.outf, epoch, whichdecoder)
    aeoutf_tran = "{}/{}_output_decoder_{}_tran.txt".format(args.outf, epoch, whichdecoder)
    with open(aeoutf_from, 'w') as f_from, open(aeoutf_tran, 'w') as f_trans:
        # laplacian smoothing
        # for word in corpus.dictionary.word2idx.keys():
        #    f_from.write(word + "\n")
        #    f_trans.write(word + "\n")
        for i in range(len(all_source_sents)):
            # real sentence
            f_from.write(all_source_sents[i])
            # transfer sentence
            f_trans.write(all_transfer_sents[i])
            if i != len(all_source_sents) - 1:
                f_from.write("\n")
                f_trans.write("\n")

    # bleu
    all_bleu_scores = 0.0
    for i in range(len(all_source_sents)):
        sou = all_source_sents[i].split(' ')
        tran = all_transfer_sents[i].split(' ')
        all_bleu_scores += sentence_bleu([sou], tran,smoothing_function=SmoothingFunction().method7,weights=[1.0/3.0]*3)
    bleu = all_bleu_scores / n_sents * 100.0

    # forward and reverse
    loss = total_loss.item() / len(data_source)
    ppl = math.exp(loss)

    #print('bleu {:4.2f} | ppl {:4.3f}'.format(bleu, ppl))
    #logging.info('bleu {:4.2f} | ppl {:4.3f}'.format(bleu, ppl))

    # transfer
    labels = fasttext_classifier.predict(all_transfer_sents)
    truth = str(1 - whichdecoder)
    transfer = float(sum([l == truth for ll in labels for l in ll])) / n_sents * 100.0

    # load sentences to evaluate on
    arpa_path = '{}/{}_lm_{}.arpa'.format(args.outf, epoch, whichdecoder)
    kenlm_model = train_ngram_lm(args.kenlm_path, aeoutf_from, arpa_path, args.N)
    forward = get_ppl(kenlm_model, all_transfer_sents)

    kenlm_model = train_ngram_lm(args.kenlm_path, aeoutf_tran, arpa_path, args.N)
    reverse = get_ppl(kenlm_model, all_source_sents)

    #print('transfer {:4.2f} | forward {:4.3f} | reverse {:4.3f}'.format(transfer, forward, reverse))
    #logging.info('transfer {:4.2f} | forward {:4.3f} | reverse {:4.3f}'.format(transfer, forward, reverse))

    return bleu, ppl, transfer, forward, reverse


def train_lm(whichdecoder, eval_path, save_path):
    # generate examples
    indices = []
    noise = to_gpu(use_cuda, Variable(torch.ones(100, args.z_size)))
    for i in range(1000):
        noise.data.normal_(0, 1)

        fake_hidden = gan_gen(noise)
        max_indices = autoencoder.generate(whichdecoder, fake_hidden, args.maxlen)
        indices.append(max_indices.data.cpu().numpy())

    indices = np.concatenate(indices, axis=0)

    sentences_gen = []
    # write generated sentences to text file
    with open(save_path + ".txt", "w") as f:
        # laplacian smoothing
        for word in corpus.dictionary.word2idx.keys():
            f.write(word + "\n")
        for idx in indices:
            # generated sentence
            words = [corpus.dictionary.idx2word[x] for x in idx]
            # truncate sentences to first occurrence of <eos>
            truncated_sent = []
            for w in words:
                if w != '<eos>':
                    truncated_sent.append(w)
                else:
                    break
            chars = " ".join(truncated_sent)
            sentences_gen.append(chars)
            f.write(chars + "\n")

    # train language model on generated examples
    lm = train_ngram_lm(kenlm_path=args.kenlm_path,
                        data_path=save_path + ".txt",
                        output_path=save_path + ".arpa",
                        N=args.N)

    # load sentences to evaluate on
    eval_path += '0' if whichdecoder == 0 else '1'
    with open(eval_path + ".txt", 'r') as f:
        lines = f.readlines()
    sentences = [l.replace('\n', '') for l in lines]
    reverse_ppl = get_ppl(lm, sentences)

    # forward
    lm = train_ngram_lm(kenlm_path=args.kenlm_path,
                        data_path=eval_path + ".txt",
                        output_path=eval_path + ".arpa",
                        N=args.N)
    forward_ppl = get_ppl(lm, sentences_gen)

    return forward_ppl, reverse_ppl


def load_models(load_path, epoch, twodecoders=False):
    model_args = json.load(open("{}/args.json".format(load_path), "r"))
    word2idx = json.load(open("{}/vocab.json".format(load_path), "r"))
    idx2word = {v: k for k, v in word2idx.items()}

    if not twodecoders:
        autoencoder = Seq2Seq(emsize=model_args['emsize'],
                              nhidden=model_args['nhidden'],
                              ntokens=model_args['ntokens'],
                              nlayers=model_args['nlayers'],
                              noise_r=model_args['noise_r'],
                              hidden_init=model_args['hidden_init'],
                              dropout=model_args['dropout'],
                              gpu=use_cuda)
    else:
        autoencoder = Seq2Seq2Decoder(emsize=model_args['emsize'],
                                      nhidden=model_args['nhidden'],
                                      ntokens=model_args['ntokens'],
                                      nlayers=model_args['nlayers'],
                                      noise_r=model_args['noise_r'],
                                      hidden_init=model_args['hidden_init'],
                                      dropout=model_args['dropout'],
                                      gpu=use_cuda)

    gan_gen = MLP_G(ninput=model_args['z_size'],
                    noutput=model_args['nhidden'],
                    layers=model_args['arch_g'])
    gan_disc = MLP_D(ninput=model_args['nhidden'],
                     noutput=1,
                     layers=model_args['arch_d'])
    classifier = MLP_Classify(ninput=args.nhidden, noutput=1, layers=args.arch_classify)

    print('Loading models from' + load_path)
    ae_path = os.path.join(load_path, "autoencoder_model_{:02d}.pt".format(epoch))
    gen_path = os.path.join(load_path, "gan_gen_model_{:02d}.pt".format(epoch))
    disc_path = os.path.join(load_path, "gan_disc_model_{:02d}.pt".format(epoch))
    classifier_path = os.path.join(load_path, "classifier_model_{:02d}.pt".format(epoch))

    autoencoder.load_state_dict(torch.load(ae_path))
    gan_gen.load_state_dict(torch.load(gen_path))
    gan_disc.load_state_dict(torch.load(disc_path))
    classifier.load_state_dict(torch.load(classifier_path))
    return autoencoder, gan_gen, gan_disc, classifier


def save_model(epoch):
    print("Saving models")
    with open('{}/autoencoder_model_{:02d}.pt'.format(args.outf, epoch), 'wb') as f:
        torch.save(autoencoder.state_dict(), f)
    with open('{}/gan_gen_model_{:02d}.pt'.format(args.outf, epoch), 'wb') as f:
        torch.save(gan_gen.state_dict(), f)
    with open('{}/gan_disc_model_{:02d}.pt'.format(args.outf, epoch), 'wb') as f:
        torch.save(gan_disc.state_dict(), f)
    with open('{}/classifier_model_{:02d}.pt'.format(args.outf, epoch), 'wb') as f:
        torch.save(classifier.state_dict(), f)


# args.load_path = './output/ARAE'
# args.load_epoch = 25
# args.eval = True
if args.load_path != '':
    autoencoder, gan_gen, gan_disc, classifier = load_models(args.load_path, args.load_epoch, twodecoders=True)
    if use_cuda:
        autoencoder = autoencoder.cuda()
        gan_gen = gan_gen.cuda()
        gan_disc = gan_disc.cuda()
        classifier = classifier.cuda()

if args.eval:
    logging.info('------------------------------------------------------')
    logging.info("evaluation:")
    epoch = args.load_epoch if args.load_path != '' else -1
    # whichdecoder, noise, epoch = 0, fixed_noise, epoch
    # evaluate_generator(0, fixed_noise, "end_of_epoch_{}".format(epoch))
    # evaluate_generator(1, fixed_noise, "end_of_epoch_{}".format(epoch))
    # whichdecoder, eval_path, save_path = 0, './data/test', '{}/epo{}_lm_{}'.format(args.outf, epoch, whichdecoder)
    # forward_ppl0, reverse_ppl0 = train_lm(0, './data/test', '{}/epo{}_lm_{}'.format(args.outf, epoch, 0))
    # forward_ppl1, reverse_ppl1 = train_lm(1, './data/test', '{}/epo{}_lm_{}'.format(args.outf, epoch, 1))
    #print('forward_ppl0 {:4.2f} | reverse_ppl0 {:4.3f}'.format(forward_ppl0, reverse_ppl0))
    #logging.info('forward_ppl0 {:4.2f} | reverse_ppl0 {:4.3f}'.format(forward_ppl0, reverse_ppl0))
    #print('forward_ppl1 {:4.2f} | reverse_ppl1 {:4.3f}'.format(forward_ppl1, reverse_ppl1))
    #logging.info('forward_ppl1 {:4.2f} | reverse_ppl1 {:4.3f}'.format(forward_ppl1, reverse_ppl1))
    # forward_ppl, reverse_ppl = (forward_ppl0 + forward_ppl1) / 2, (reverse_ppl0 + reverse_ppl1) / 2
    # print('forward_ppl {:4.2f} | reverse_ppl {:4.3f}'.format(forward_ppl, reverse_ppl))
    # logging.info('forward_ppl {:4.2f} | reverse_ppl {:4.3f}'.format(forward_ppl, reverse_ppl))
    # whichdecoder, data_source, epoch = 0, test0_data, epoch
    bleu0, ppl0, transfer0, forward0, reverse0 = evaluate_autoencoder(0, test0_data, epoch)
    bleu1, ppl1, transfer1, forward1, reverse1 = evaluate_autoencoder(1, test1_data, epoch)
    bleu, ppl = (bleu0 + bleu1) / 2, (ppl0 + ppl1) / 2
    #print('bleu {:4.2f} | ppl {:4.3f}'.format(bleu, ppl))
    #logging.info('bleu {:4.2f} | ppl {:4.3f}'.format(bleu, ppl))
    transfer, forward, reverse = (transfer0 + transfer1) / 2, (forward0 + forward1) / 2, (reverse0 + reverse1) / 2
    #print('transfer {:4.2f} | forward {:4.2f} | reverse {:4.3f}'.format(transfer, forward, reverse))
    #logging.info('transfer {:4.2f} | forward {:4.2f} | reverse {:4.3f}'.format(transfer, forward, reverse))
    print('transfer {:4.2f} | bleu {:4.2f} | forward {:4.2f} | reverse {:4.3f}'.format(transfer, bleu, ppl, reverse))
    logging.info('transfer {:4.2f} | bleu {:4.2f} | forward {:4.2f} | reverse {:4.3f}'.format(transfer, bleu, ppl, reverse))
    exit()

###############################################################################
# Training code
###############################################################################


def train_ae(whichdecoder, batch):
    autoencoder.train()
    optimizer_ae.zero_grad()

    source, target, lengths = batch
    source = to_gpu(use_cuda, Variable(source))
    target = to_gpu(use_cuda, Variable(target))

    mask = target.gt(0)
    masked_target = target.masked_select(mask)
    output_mask = mask.unsqueeze(1).expand(mask.size(0), ntokens)
    output = autoencoder(whichdecoder, source, lengths, noise=True)
    flat_output = output.view(-1, ntokens)
    masked_output = flat_output.masked_select(output_mask).view(-1, ntokens)
    loss = criterion_ce(masked_output / args.temp, masked_target)
    loss.backward()

    # `clip_grad_norm` to prevent exploding gradient in RNNs / LSTMs
    torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), args.clip)
    optimizer_ae.step()

    return loss


def train_gan_d_into_ae(whichdecoder, batch):
    autoencoder.train()
    optimizer_ae.zero_grad()

    source, target, lengths = batch
    source = to_gpu(use_cuda, Variable(source))
    real_hidden = autoencoder(whichdecoder, source, lengths, noise=False, encode_only=True)
    real_hidden.register_hook(grad_hook)
    errD_real = gan_disc(real_hidden)
    errD_real.backward(one)
    torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), args.clip)

    optimizer_ae.step()

    return errD_real


def grad_hook_cla(grad):
    return grad * args.lambda_class


def classifier_regularize(whichclass, batch):
    autoencoder.train()
    autoencoder.zero_grad()

    source, target, lengths = batch
    source = to_gpu(use_cuda, Variable(source))
    flippedclass = abs(1 - whichclass)
    labels = to_gpu(use_cuda, Variable(torch.zeros(source.size(0)).fill_(flippedclass)))

    # Train
    code = autoencoder(0, source, lengths, noise=False, encode_only=True)
    code.register_hook(grad_hook_cla)
    scores = classifier(code)
    classify_reg_loss = F.binary_cross_entropy(scores.squeeze(1), labels)
    classify_reg_loss.backward()

    torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), args.clip)
    optimizer_ae.step()

    return classify_reg_loss


def train_gan_g():
    gan_gen.train()
    gan_gen.zero_grad()

    noise = to_gpu(use_cuda, Variable(torch.ones(args.batch_size, args.z_size)))
    noise.data.normal_(0, 1)
    fake_hidden = gan_gen(noise)
    errG = gan_disc(fake_hidden)
    errG.backward(mone)
    optimizer_gan_g.step()

    return errG


def grad_hook(grad):
    return grad * args.grad_lambda


def train_classifier(whichclass, batch):
    classifier.train()
    classifier.zero_grad()

    source, target, lengths = batch
    source = to_gpu(use_cuda, Variable(source))
    labels = to_gpu(use_cuda, Variable(torch.zeros(source.size(0)).fill_(whichclass)))

    # Train
    code = autoencoder(0, source, lengths, noise=False, encode_only=True).detach()
    scores = classifier(code)
    classify_loss = F.binary_cross_entropy(scores.squeeze(1), labels)
    classify_loss.backward()
    optimizer_classify.step()
    classify_loss = classify_loss.cpu().item()

    pred = scores.data.round().squeeze(1)
    accuracy = pred.eq(labels.data).float().mean()

    return classify_loss, accuracy


# Steal from https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
def calc_gradient_penalty(netD, real_data, fake_data):
    bsz = real_data.size(0)
    alpha = torch.rand(bsz, 1)
    alpha = alpha.expand(bsz, real_data.size(1))  # only works for 2D XXX
    alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * args.gan_gp_lambda
    return gradient_penalty


def train_gan_d(whichdecoder, batch):
    gan_disc.train()
    optimizer_gan_d.zero_grad()

    # positive samples ----------------------------
    # generate real codes
    source, target, lengths = batch
    source = to_gpu(use_cuda, Variable(source))

    # batch_size x nhidden
    real_hidden = autoencoder(whichdecoder, source, lengths, noise=False, encode_only=True)

    # loss / backprop
    errD_real = gan_disc(real_hidden)
    errD_real.backward(mone)

    # negative samples ----------------------------
    # generate fake codes
    noise = to_gpu(use_cuda, Variable(torch.ones(args.batch_size, args.z_size)))
    noise.data.normal_(0, 1)

    # loss / backprop
    fake_hidden = gan_gen(noise)
    errD_fake = gan_disc(fake_hidden.detach())
    errD_fake.backward(one)

    # gradient penalty
    gradient_penalty = calc_gradient_penalty(gan_disc, real_hidden.data, fake_hidden.data)
    gradient_penalty.backward()

    optimizer_gan_d.step()
    errD = errD_fake - errD_real

    return errD, errD_real, errD_fake


logging.info('------------------------------------------------------')
logging.info("Training...")
start_epoch = 0 if args.load_path == '' else args.load_epoch

for epoch in range(start_epoch + 1, args.epochs + 1):
    logging.info('------------------------------------------------------')
    logging.info("the current epo is %d of %d" % (epoch, args.epochs))
    print("the current epo is %d of %d" % (epoch, args.epochs))
    logging.info("training:")
    print("training:")

    # shuffle between epochs
    train0_data = batchify(corpus.data['train0'], args.batch_size, shuffle=True)
    train1_data = batchify(corpus.data['train1'], args.batch_size, shuffle=True)

    # loop through all batches in training data
    niter_global = 0
    pbar = tqdm(range(max(len(train0_data), len(train1_data))))
    for niter in pbar:
        niter0 = niter if niter < len(train0_data) else random.randint(0, len(train0_data) - 1)
        niter1 = niter if niter < len(train1_data) else random.randint(0, len(train1_data) - 1)

        # train autoencoder ----------------------------
        # whichdecoder, batch = 0, train0_data[niter0]
        train_ae(0, train0_data[niter0])
        train_ae(1, train1_data[niter1])

        # train classifier ----------------------------
        train_classifier(0, train0_data[niter0])
        train_classifier(1, train1_data[niter1])

        # classifier regularize to autoencoder
        # whichclass, batch = 0, train0_data[niter0]
        classifier_regularize(0, train0_data[niter0])
        classifier_regularize(1, train1_data[niter1])

        # train gan ----------------------------------
        # train discriminator/critic
        for i in range(args.niters_gan_d):
            batch = train0_data[niter0] if i % 2 == 0 else train1_data[niter1]
            whichdecoder = 0 if i % 2 == 0 else 1
            train_gan_d(whichdecoder, batch)

        # train generator
        train_gan_g()

        # train autoencoder from d
        train_gan_d_into_ae(0, train0_data[niter0])
        train_gan_d_into_ae(1, train1_data[niter1])

        niter_global += 1
        if niter_global % 100 == 0:
            # exponentially decaying noise on autoencoder
            autoencoder.noise_r = autoencoder.noise_r * args.noise_anneal

    save_model(epoch)
