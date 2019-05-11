
import sys
import os
from models_yelp import Encoder_AE, Decoder
import torch
from torch import optim
from itertools import chain
import argparse
import logging
import torch.nn as nn
import numpy as np
import math
from tqdm import tqdm
from data import MonoTextData

parser = argparse.ArgumentParser()

# global parameters
parser.add_argument('--train_data', default='data/yelp.train.txt')
parser.add_argument('--val_data', default='data/yelp.valid.txt')
parser.add_argument('--test_data', default='data/yelp.test.txt')
parser.add_argument('--results_folder_prefix', default='results_')
parser.add_argument('--train_from', default='')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--test', action="store_true")
parser.add_argument('--log_prefix', default='eval')
parser.add_argument('--model', default='ae', type=str)

# global training parameters
parser.add_argument('--num_epochs', default=40, type=int)
parser.add_argument('--batch_size', default=32, type=int)

# use GPU
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--no_gpu', action="store_true")

# model and optimizer parameters
parser.add_argument('--latent_dim', default=32, type=int)
parser.add_argument('--enc_word_dim', default=512, type=int)
parser.add_argument('--enc_h_dim', default=1024, type=int)
parser.add_argument('--enc_num_layers', default=1, type=int)

parser.add_argument('--dec_word_dim', default=512, type=int)
parser.add_argument('--dec_h_dim', default=1024, type=int)
parser.add_argument('--dec_num_layers', default=1, type=int)
parser.add_argument('--dec_dropout', default=0.5, type=float)

parser.add_argument('--end2end_lr', default=8e-4, type=float)
parser.add_argument('--max_grad_norm', default=5.0, type=float)

if sys.argv[1:] == ['0', '0']:
    args = parser.parse_args([])   # run in pycharm console
else:
    args = parser.parse_args()  # run in cmd

# parameters
train_data_all = MonoTextData(args.train_data, label=True)
vocab = train_data_all.vocab
vocab_size = len(vocab)
val_data_all = MonoTextData(args.val_data, label=True, vocab=vocab)
test_data_all = MonoTextData(args.test_data, label=True, vocab=vocab)
print('Batch size: %d' % args.batch_size)
print('Train data: %d sentences' % len(train_data_all))
print('Val data: %d sentences' % len(val_data_all))
print('Test data: %d sentences' % len(test_data_all))
print('finish reading datasets, vocab size is %d' % len(vocab))
print('dropped sentences: %d' % train_data_all.dropped)

results_folder = args.results_folder_prefix + args.model + '/'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
logging.basicConfig(filename=os.path.join(results_folder, args.log_prefix+'.log'),
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

device = torch.device("cuda" if gpu else "cpu")
train_data = train_data_all.create_data_batch(batch_size=args.batch_size, device=device, batch_first=True)
# val_data_batch = val_data_all.create_data_batch(batch_size=args.batch_size, device=device, batch_first=True)
test_data = test_data_all.create_data_batch(batch_size=args.batch_size, device=device, batch_first=True)

epo_0 = 0

encoder = Encoder_AE(vocab_size=vocab_size,
                  enc_word_dim=args.enc_word_dim,
                  enc_h_dim=args.enc_h_dim,
                  enc_num_layers=args.enc_num_layers,
                  latent_dim=args.latent_dim)
decoder = Decoder(vocab_size=vocab_size,
                  dec_word_dim=args.dec_word_dim,
                  dec_h_dim=args.dec_h_dim,
                  dec_num_layers=args.dec_num_layers,
                  dec_dropout=args.dec_dropout,
                  latent_dim=args.latent_dim)
criterion = nn.NLLLoss()
end2end_optimizer = optim.Adam(chain(encoder.parameters(), decoder.parameters()), lr=args.end2end_lr)

if args.train_from == "":
    for param in encoder.parameters():
        param.data.uniform_(-0.01, 0.01)
    for param in decoder.parameters():
        param.data.uniform_(-0.01, 0.01)
    if gpu:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        criterion.cuda()
else:
    logging.info('load model from' + args.train_from)
    checkpoint = torch.load(args.train_from, map_location="cuda:" + str(args.gpu) if gpu else 'cpu')

    if not args.test:
        np_random_state = checkpoint['np_random_state']
        prng.set_state(np_random_state)
        torch_rng_state = checkpoint['torch_rng_state']
        torch_rng_state_cuda = checkpoint['torch_rng_state_cuda']
        torch.set_rng_state(torch_rng_state.cpu())
        if gpu: torch.cuda.set_rng_state(torch_rng_state_cuda.cpu())

    encoder = checkpoint['encoder']
    decoder = checkpoint['decoder']
    criterion = checkpoint['criterion']
    end2end_optimizer = checkpoint['end2end_optimizer']

    epo_0 = int(args.train_from[-6:-3])

logging.info("model configuration:")
logging.info(str(encoder))
logging.info(str(decoder))


def evaluation(data):
    encoder.eval()
    decoder.dec_linear.eval()
    decoder.dropout.eval()

    num_sents = 0.0
    num_words = 0.0
    total_rec = 0.0
    total_mean_au = torch.zeros(args.latent_dim, device='cuda' if gpu else 'cpu')
    total_sq_au = torch.zeros(args.latent_dim, device='cuda' if gpu else 'cpu')

    pbar = tqdm(range(len(data)))
    for mini_batch in pbar:
        # logging.info('batch: %d' % mini_batch)

        sents = data[mini_batch]
        batch_size, length = sents.size()
        # not predict start symbol
        length -= 1
        num_sents += batch_size
        num_words += batch_size * length
        if gpu: sents = sents.cuda()

        z_x = encoder(sents)
        z_x = z_x.data

        # rec, kl
        preds = decoder(sents, z_x).data
        rec = sum([criterion(preds[:, l], sents[:, l + 1]) for l in range(preds.size(1))])
        total_rec += rec.item() * batch_size

        # active units
        total_mean_au += torch.sum(z_x, dim=0)
        total_sq_au += torch.sum(z_x ** 2, dim=0)

        del z_x
        torch.cuda.empty_cache()

    rec = total_rec / num_sents
    ppl = math.exp(total_rec / num_words)

    logging.info('rec: %.4f' % rec)
    logging.info('ppl: %.4f' % ppl)

    mean_au = total_mean_au / num_sents
    sq_au = total_sq_au / num_sents
    au_cov = sq_au - mean_au ** 2
    au = (au_cov >= 0.01).sum().item()
    logging.info('au_cov: %s' % str(au_cov))
    logging.info('au: %.4f' % au)

    report = "rec %f, ppl %f, au %f\n" % (rec, ppl, au)
    print(report)

    encoder.train()
    decoder.train()

    return rec


def sample_sentences(decoder, vocab, num_sentences, reconstruction=False, data=test_data):
    logging.info('---------------- Sample sentences: ----------------')
    decoder.eval()
    sampled_sents = []

    if reconstruction:
        sample_batch = torch.randint(len(data), (1,))
        sents_batch = data[sample_batch]
        batch_size = sents_batch.size()[0]
        if gpu: sents_batch = sents_batch.cuda()
        eps = torch.randn((batch_size, args.latent_dim), device=sents_batch.device)
        z_x, _ = encoder(sents_batch, eps)
        expand_int = torch.randint(z_x.shape[0], (num_sentences,)).tolist()
        z_x = z_x.data[expand_int, :]
        sents = sents_batch.data[expand_int, :].tolist()
        sents = [[vocab.idx2word[s] for s in sents[i]] for i in range(num_sentences)]
    else:
        z_x = torch.randn((num_sentences, args.latent_dim), device='cuda' if gpu else 'cpu')

    for i in range(num_sentences):
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
        if reconstruction:
            logging.info(('the %d-th real sent: ') % i + ' '.join(sents[i]))
        logging.info(('the %d-th fake sent: ') % i + ' '.join(sent))


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


def check_point(epo=None):
    check_pt = {
        'args': args,
        'encoder': encoder,
        'decoder': decoder,
        'criterion': criterion,
        'end2end_optimizer': end2end_optimizer,

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
    evaluation(test_data)
    # sample_sentences(decoder, vocab, num_sentences=50, reconstruction=False, data=test_data)
    # sample_sentences(decoder, vocab, num_sentences=50, reconstruction=True, data=test_data)
    exit()

logging.info('\n------------------------------------------------------')
logging.info("the current epo is %d of %d" % (epo_0, args.num_epochs))
print("the current epo is %d of %d" % (epo_0, args.num_epochs))
logging.info("evaluation:")
print("evaluation:")
check_point(epo_0)
# evaluation(test_data)
# sample_sentences(decoder, vocab, num_sentences=50, reconstruction=False, data=test_data)
# sample_sentences(decoder, vocab, num_sentences=50, reconstruction=True, data=test_data)

for epo in torch.arange(epo_0 + 1, args.num_epochs + 1):
    logging.info('\n------------------------------------------------------')
    logging.info("the current epo is %d of %d" % (epo, args.num_epochs))
    print("the current epo is %d of %d" % (epo, args.num_epochs))
    logging.info("training:")
    print("training:")

    # training
    encoder.train()
    decoder.train()

    random_bat = torch.randperm(len(train_data)).tolist()
    pbar = tqdm(range(len(train_data)))
    for bat in pbar:
        mini_batch = random_bat[bat]
        sents = train_data[mini_batch]
        batch_size, length = sents.size()
        # not predict start symbol
        length -= 1

        if gpu: sents = sents.cuda()
        z_x = encoder(sents)

        # end2end update
        preds = decoder(sents, z_x)
        rec = sum([criterion(preds[:, l], sents[:, l + 1]) for l in range(preds.size(1))])
        loss = rec
        end2end_optimizer.zero_grad()
        loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.max_grad_norm)
        end2end_optimizer.step()

        del loss
        torch.cuda.empty_cache()
        assert not torch.isnan(z_x).any(), 'training get nan z_x'

    # evaluation
    logging.info("evaluation:")
    print("evaluation:")
    check_point(epo)
    if epo >= 30: evaluation(test_data)
    # sample_sentences(decoder, vocab, num_sentences=50, reconstruction=False, data=test_data)
    # sample_sentences(decoder, vocab, num_sentences=50, reconstruction=True, data=test_data)
