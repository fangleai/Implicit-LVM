
import argparse
from datetime import datetime
import numpy as np
import logging
import torch
import os, sys
from utils_dailydial import indexes2sent, gData, gVar, Metrics
from models_dailydial import ILVM
from data import DailyDialCorpus, DailyDialDataLoader
from tqdm import tqdm

config = {
    'maxlen': 40,  # maximum utterance length
    'diaglen': 10,  # how many utterance kept in the context window

    # Model Arguments
    'emb_size': 200,  # size of word embeddings
    'n_hidden': 300,  # number of hidden units per layer
    'n_layers': 1,  # number of layers
    'noise_radius': 0.2,  # stdev of noise for autoencoder (regularizer)
    'z_size': 200,  # dimension of z # 300 performs worse
    'temp': 1.0,  # softmax temperature (lower --> more discrete)
    'dropout': 0.5,  # dropout applied to layers (0 = no dropout)

    # Training Arguments
    'batch_size': 32,
    'epochs': 100,  # maximum number of epochs
    'min_epochs': 2,  # minimum number of epochs to train for

    'n_iters_d': 5,  # number of discriminator iterations in training
    'lr_end2end_lstm': 1.0,  # end2end learning rate for lstm
    'lr_end2end_fc': 5e-5,  # end2end learning rate for fc
    'lr_nu': 1e-5,  # nu_net learning rate
    'beta1': 0.9,  # beta1 for adam
    'clip': 1.0,  # gradient clipping, max norm
    'gan_clamp': 0.01,  # WGAN clamp (Do not use clamp when you apply gradient penelty
}
parser = argparse.ArgumentParser()

# Global Arguments
parser.add_argument('--data_path', type=str, default='./data/', help='location of the data corpus')
parser.add_argument('--model', type=str, default='ILVM', help='model name')
parser.add_argument('--expname', type=str, default='basic',
                    help='experiment name, for disinguishing different parameter settings')
parser.add_argument('--visual', action='store_true', default=False, help='visualize training status in tensorboard')
parser.add_argument('--reload_from', type=int, default=-1, help='reload from a trained ephoch')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--no_gpu', action="store_true")

# # KL cost annealing, increase beta from beta_0 by 1/warmup in certain steps
# parser.add_argument('--warmup', default=10, type=int)
# parser.add_argument('--beta_0', default=0.1, type=float)

# Evaluation Arguments
parser.add_argument('--eval', action='store_true', help='evaluation')
parser.add_argument('--log_prefix', default='')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--n_samples', type=int, default=10, help='Number of responses to sampling')

if sys.argv[1:] == ['0', '0']:
    args = parser.parse_args([])   # run in pycharm console
else:
    args = parser.parse_args()  # run in cmd

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
prng = np.random.RandomState()
torch.manual_seed(args.seed)
if not torch.cuda.is_available(): args.no_gpu = True
use_cuda = not args.no_gpu
if use_cuda:
    torch.cuda.set_device(args.gpu_id)  # set gpu device
    torch.cuda.manual_seed(args.seed)

# Load data
corpus = DailyDialCorpus(args.data_path, wordvec_path=args.data_path+'glove.twitter.27B.200d.txt', wordvec_dim=200)
dials = corpus.get_dialogs()
metas = corpus.get_metas()
train_dial, valid_dial, test_dial = dials.get("train"), dials.get("valid"), dials.get("test")
train_meta, valid_meta, test_meta = metas.get("train"), metas.get("valid"), metas.get("test")
train_loader = DailyDialDataLoader("Train", train_dial, train_meta, config['maxlen'])
valid_loader = DailyDialDataLoader("Valid", valid_dial, valid_meta, config['maxlen'])
test_loader = DailyDialDataLoader("Test", test_dial, test_meta, config['maxlen'])

vocab = corpus.vocab
ivocab = corpus.ivocab
n_tokens = len(vocab)
metrics = Metrics(corpus.word2vec)

print("Loaded data!")

# make output directory if it doesn't already exist
if not os.path.isdir('./output'):
    os.makedirs('./output')
if not os.path.isdir('./output/{}'.format(args.model)):
    os.makedirs('./output/{}'.format(args.model))
logging.basicConfig(filename=os.path.join('./output/', args.model, args.log_prefix + 'logs.txt'),
                    level=logging.INFO, format='%(asctime)s--- %(message)s')
logging.info("the configuration:")
logging.info(str(args).replace(',', '\n'))
logging.info(str(config).replace(',', '\n'))

# Define the models
model = ILVM(config, n_tokens)
if use_cuda: model = model.cuda()

if args.reload_from > 0:
    logging.info('Loading models')
    model = torch.load(f='./output/{}/model_epo{}.pckl'.format(args.model, args.reload_from),
                       map_location="cuda:" + str(args.gpu_id) if use_cuda else 'cpu')

if corpus.word2vec is not None and args.reload_from < 0:
    print("Loaded word2vec")
    model.embedder.weight.data.copy_(torch.from_numpy(corpus.word2vec))
    model.embedder.weight.data[0].fill_(0)

logging.info("model configuration:")
logging.info(str(model))


def evaluate(model, metrics, test_loader, ivocab, vocab, repeat, PAD_token=0):
    recall_bleus, prec_bleus, bows_extrema, bows_avg, bows_greedy, intra_dist1s, intra_dist2s, \
    avg_lens, inter_dist1s, inter_dist2s = [], [], [], [], [], [], [], [], [], []
    local_t = 0

    model.eval()
    pbar = tqdm(range(test_loader.num_batch))

    for bat in pbar:
        batch = test_loader.next_batch()
        if bat == test_loader.num_batch: break  # end of epoch

        local_t += 1

        context, context_lens, utt_lens, floors, _, _, _, response, res_lens, _ = batch
        # remove the sos token in the context and reduce the context length
        context, utt_lens = context[:, :, 1:], utt_lens - 1

        if local_t % 2000 == 0:
            logging.info("Batch %d \n" % (local_t))  # print the context

        start = np.maximum(0, context_lens[0] - 5)
        for t_id in range(start, context.shape[1], 1):
            context_str = indexes2sent(context[0, t_id], ivocab, ivocab["</s>"], PAD_token)
            if local_t % 2000 == 0:
                logging.info("Context %d-%d: %s\n" % (t_id, floors[0, t_id], context_str))
        # print the true outputs
        ref_str, _ = indexes2sent(response[0], ivocab, ivocab["</s>"], ivocab["<s>"])
        ref_tokens = ref_str.split(' ')

        if local_t % 2000 == 0:
            logging.info("Target >> %s\n" % (ref_str.replace(" ' ", "'")))

        context, context_lens, utt_lens, floors = gVar(context), gVar(context_lens), gVar(utt_lens), gData(floors)
        sample_words, sample_lens = model.sample(context, context_lens, utt_lens, floors, repeat, ivocab["<s>"],
                                                 ivocab["</s>"])
        # nparray: [repeat x seq_len]
        pred_sents, _ = indexes2sent(sample_words, ivocab, ivocab["</s>"], PAD_token)
        pred_tokens = [sent.split(' ') for sent in pred_sents]
        for r_id, pred_sent in enumerate(pred_sents):
            if local_t % 2000 == 0:
                logging.info("Sample %d >> %s\n" % (r_id, pred_sent.replace(" ' ", "'")))

        max_bleu, avg_bleu = metrics.sim_bleu(pred_tokens, ref_tokens)
        recall_bleus.append(max_bleu)
        prec_bleus.append(avg_bleu)

        bow_extrema, bow_avg, bow_greedy = metrics.sim_bow(sample_words, sample_lens, response[:, 1:], res_lens - 2)
        bows_extrema.append(bow_extrema)
        bows_avg.append(bow_avg)
        bows_greedy.append(bow_greedy)

        intra_dist1, intra_dist2, inter_dist1, inter_dist2 = metrics.div_distinct(sample_words, sample_lens)
        intra_dist1s.append(intra_dist1)
        intra_dist2s.append(intra_dist2)
        avg_lens.append(np.mean(sample_lens))
        inter_dist1s.append(inter_dist1)
        inter_dist2s.append(inter_dist2)

        # logging.info("\n")

    recall_bleu = float(np.mean(recall_bleus))
    prec_bleu = float(np.mean(prec_bleus))
    f1 = 2 * (prec_bleu * recall_bleu) / (prec_bleu + recall_bleu + 10e-12)

    bow_extrema = float(np.mean(bows_extrema))
    bow_avg = float(np.mean(bows_avg))
    bow_greedy = float(np.mean(bows_greedy))

    intra_dist1 = float(np.mean(intra_dist1s))
    intra_dist2 = float(np.mean(intra_dist2s))

    avg_len = float(np.mean(avg_lens))

    inter_dist1 = float(np.mean(inter_dist1s))
    inter_dist2 = float(np.mean(inter_dist2s))

    report = "Avg recall BLEU %f, avg precision BLEU %f, F1 %f, \nbow_avg %f, bow_extrema %f, bow_greedy %f, \n" \
             "intra_dist1 %f, intra_dist2 %f, inter_dist1 %f, inter_dist2 %f, \navg_len %f" \
             % (recall_bleu, prec_bleu, f1, bow_avg, bow_extrema, bow_greedy, intra_dist1, intra_dist2,
                inter_dist1, inter_dist2, avg_len)
    print(report)
    logging.info(report + "\n")
    print("Done testing")

    model.train()

    return recall_bleu,prec_bleu,bow_extrema,bow_avg,bow_greedy,intra_dist1,intra_dist2,avg_len,inter_dist1,inter_dist2


def save_model(model, epoch):
    torch.save(f='./output/{}/model_epo{}.pckl'.format(args.model, epoch), obj=model)


# args.eval = True
if args.eval:
    logging.info('------------------------------------------------------')
    logging.info("evaluation:")
    test_loader.epoch_init(1, config['diaglen'], 1, shuffle=False)
    evaluate(model, metrics, test_loader, ivocab, vocab, repeat=args.n_samples, PAD_token=0)
    exit()

logging.info('------------------------------------------------------')
logging.info("Training...")
start_epoch = 0 if args.reload_from == -1 else args.reload_from
# beta = args.beta_0

for epoch in range(start_epoch + 1, config['epochs'] + 1):
    logging.info("the current epo is %d" % epoch)
    print("the current epo is %d" % epoch)

    model.train()

    # shuffle (re-define) data between epochs   
    train_loader.epoch_init(config['batch_size'], config['diaglen'], 1, shuffle=True)
    pbar = tqdm(range(train_loader.num_batch))

    for bat in pbar:
        batch = train_loader.next_batch()
        if bat == train_loader.num_batch: break  # end of epoch

        # if args.warmup > 0:
        #     beta = min(1.0, beta + 1. / (args.warmup * train_loader.num_batch))

        context, context_lens, utt_lens, floors, _, _, _, response, res_lens, _ = batch
        # remove the sos token in the context and reduce the context length
        context, utt_lens = context[:, :, 1:], utt_lens - 1
        context, context_lens, utt_lens, floors, response, res_lens \
            = gVar(context), gVar(context_lens), gVar(utt_lens), gData(floors), gVar(response), gVar(res_lens)
        batch_size = context.size(0)

        #   ae phase
        model.context_encoder.train()
        model.decoder.train()

        c = model.context_encoder(context, context_lens, utt_lens, floors)
        x, _ = model.utt_encoder(response[:, 1:], res_lens - 1)
        z = model.sample_code_post(x, c)

        output = model.decoder(torch.cat((z, c), 1), None, response[:, :-1], (res_lens - 1))
        flattened_output = output.view(-1, model.vocab_size)
        dec_target = response[:, 1:].contiguous().view(-1)
        mask = dec_target.gt(0)  # [(batch_sz*seq_len)]
        masked_target = dec_target.masked_select(mask)  #
        output_mask = mask.unsqueeze(1).expand(mask.size(0), model.vocab_size)  # [(batch_sz*seq_len) x n_tokens]
        masked_output = flattened_output.masked_select(output_mask).view(-1, model.vocab_size)

        model.optimizer_end2end_lstm.zero_grad()
        loss = model.criterion_ce(masked_output / model.temp, masked_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.context_encoder.parameters())
                                       + list(model.decoder.parameters()), model.clip)
        model.optimizer_end2end_lstm.step()

        del loss
        for p in model.embedder.parameters():
            if torch.isnan(p).any():
                print('training get nan embed after end2end phase')
                logging.info('training get nan embed after end2end phase')
                exit()

        # g phase
        model.context_encoder.eval()
        model.optimizer_end2end_fc.zero_grad()

        for p in model.nu_net.parameters():
            p.requires_grad = False

        c = model.context_encoder(context, context_lens, utt_lens, floors)
        x, _ = model.utt_encoder(response[:, 1:], res_lens - 1)
        x = x.data
        c = c.data
        z_x = model.sample_code_post(x, c)
        z = model.sample_code_prior(c)
        # nu_loss = beta * torch.mean(model.nu_forward(z_x, c) - torch.exp(model.nu_forward(z, c)))
        nu_loss = torch.mean(model.nu_forward(z_x, c) - torch.exp(model.nu_forward(z, c)))
        nu_loss.backward()
        model.optimizer_end2end_fc.step()

        for p in model.nu_net.parameters():
            p.requires_grad = True

        del z_x, nu_loss
        for p in model.recognition_net.parameters():
            if torch.isnan(p).any():
                print('training get nan recog after end2end phase')
                logging.info('training get nan recog after end2end phase')
                exit()

        # nu update
        model.context_encoder.eval()
        model.nu_net.train()

        for i in range(config['n_iters_d']):
            c = model.context_encoder(context, context_lens, utt_lens, floors)
            x, _ = model.utt_encoder(response[:, 1:], res_lens - 1)
            z_x = model.sample_code_post(x, c)
            z = model.sample_code_prior(c)

            model.optimizer_nu.zero_grad()
            x_nu, c_nu, z_x_nu, z_nu = x.data, c.data, z_x.data, z.data
            nu_loss = torch.mean(torch.exp(model.nu_forward(z_nu, c_nu)) - model.nu_forward(z_x_nu, c_nu))
            nu_loss.backward()
            model.optimizer_nu.step()

            del z_x, z, x, c, z_x_nu, z_nu, nu_loss
            for p in model.nu_net.parameters():
                if torch.isnan(p).any():
                    print('training get nan nu after nu phase')
                    logging.info('training get nan nu after nu phase')
                    exit()

        torch.cuda.empty_cache()

    logging.info('------------------------------------------------------')
    logging.info("the current epo is %d" % epoch)
    logging.info("evaluation:")
    model.adjust_lr()
    save_model(model, epoch)
    test_loader.epoch_init(1, config['diaglen'], 1, shuffle=False)
    evaluate(model, metrics, test_loader, ivocab, vocab, repeat=args.n_samples, PAD_token=0)
