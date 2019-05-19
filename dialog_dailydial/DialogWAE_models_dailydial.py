
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.init as weight_init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os
import numpy as np
import sys
from utils_dailydial import gVar, gData


class Encoder(nn.Module):
    def __init__(self, embedder, input_size, hidden_size, bidirectional, n_layers, noise_radius=0.2):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.noise_radius = noise_radius
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        assert type(self.bidirectional) == bool

        self.embedding = embedder
        self.rnn = nn.GRU(input_size, hidden_size, n_layers, batch_first=True, bidirectional=bidirectional)
        self.init_weights()

    def init_weights(self):
        for w in self.rnn.parameters():
            if w.dim() > 1:
                weight_init.orthogonal_(w)

    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

    def forward(self, inputs, input_lens=None, noise=False):
        if self.embedding is not None:
            inputs = self.embedding(inputs)

        batch_size, seq_len, emb_size = inputs.size()
        inputs = F.dropout(inputs, 0.5, self.training)

        if input_lens is not None:
            input_lens_sorted, indices = input_lens.sort(descending=True)
            inputs_sorted = inputs.index_select(0, indices)
            inputs = pack_padded_sequence(inputs_sorted, input_lens_sorted.data.tolist(), batch_first=True)

        init_hidden = gVar(torch.zeros(self.n_layers * (1 + self.bidirectional), batch_size, self.hidden_size))
        hids, h_n = self.rnn(inputs, init_hidden)
        if input_lens is not None:
            _, inv_indices = indices.sort()
            hids, lens = pad_packed_sequence(hids, batch_first=True)
            hids = hids.index_select(0, inv_indices)
            h_n = h_n.index_select(1, inv_indices)
        h_n = h_n.view(self.n_layers, (1 + self.bidirectional), batch_size, self.hidden_size)
        h_n = h_n[-1]
        enc = h_n.transpose(1, 0).contiguous().view(batch_size, -1)
        if noise and self.noise_radius > 0:
            gauss_noise = gVar(torch.normal(means=torch.zeros(enc.size()), std=self.noise_radius))
            enc = enc + gauss_noise

        return enc, hids


class ContextEncoder(nn.Module):
    def __init__(self, utt_encoder, input_size, hidden_size, n_layers=1, noise_radius=0.2):
        super(ContextEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.noise_radius = noise_radius

        self.n_layers = n_layers

        self.utt_encoder = utt_encoder
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.init_weights()

    def init_weights(self):
        for w in self.rnn.parameters():  # initialize the gate weights with orthogonal
            if w.dim() > 1:
                weight_init.orthogonal_(w)

    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

    def forward(self, context, context_lens, utt_lens, floors, noise=False):
        batch_size, max_context_len, max_utt_len = context.size()
        utts = context.view(-1, max_utt_len)
        utt_lens = utt_lens.view(-1)
        utt_encs, _ = self.utt_encoder(utts, utt_lens)
        utt_encs = utt_encs.view(batch_size, max_context_len, -1)

        floor_one_hot = gVar(torch.zeros(floors.numel(), 2))
        floor_one_hot.data.scatter_(1, floors.view(-1, 1), 1)
        floor_one_hot = floor_one_hot.view(-1, max_context_len, 2)
        utt_floor_encs = torch.cat([utt_encs, floor_one_hot], 2)

        utt_floor_encs = F.dropout(utt_floor_encs, 0.25, self.training)
        context_lens_sorted, indices = context_lens.sort(descending=True)
        utt_floor_encs = utt_floor_encs.index_select(0, indices)
        utt_floor_encs = pack_padded_sequence(utt_floor_encs, context_lens_sorted.data.tolist(), batch_first=True)

        init_hidden = gVar(torch.zeros(1, batch_size, self.hidden_size))
        hids, h_n = self.rnn(utt_floor_encs, init_hidden)

        _, inv_indices = indices.sort()
        h_n = h_n.index_select(1, inv_indices)

        enc = h_n.transpose(1, 0).contiguous().view(batch_size, -1)

        if noise and self.noise_radius > 0:
            gauss_noise = gVar(torch.normal(means=torch.zeros(enc.size()), std=self.noise_radius))
            enc = enc + gauss_noise
        return enc


class Variation(nn.Module):
    def __init__(self, input_size, z_size):
        super(Variation, self).__init__()
        self.input_size = input_size
        self.z_size = z_size
        self.fc = nn.Sequential(
            nn.Linear(input_size, z_size),
            nn.BatchNorm1d(z_size, eps=1e-05, momentum=0.1),
            nn.Tanh(),
            nn.Linear(z_size, z_size),
            nn.BatchNorm1d(z_size, eps=1e-05, momentum=0.1),
            nn.Tanh(),
        )
        self.context_to_mu = nn.Linear(z_size, z_size)  # activation???
        self.context_to_logsigma = nn.Linear(z_size, z_size)

        self.fc.apply(self.init_weights)
        self.init_weights(self.context_to_mu)
        self.init_weights(self.context_to_logsigma)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.uniform_(-0.02, 0.02)
            m.bias.data.fill_(0)

    def forward(self, context):
        batch_size, _ = context.size()
        context = self.fc(context)
        mu = self.context_to_mu(context)
        logsigma = self.context_to_logsigma(context)
        std = torch.exp(0.5 * logsigma)

        epsilon = gVar(torch.randn([batch_size, self.z_size]))
        z = epsilon * std + mu
        return z, mu, logsigma


class Decoder(nn.Module):
    def __init__(self, embedder, input_size, hidden_size, vocab_size, n_layers=1):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embedding = embedder
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        for w in self.rnn.parameters():
            if w.dim() > 1:
                weight_init.orthogonal_(w)
        self.out.weight.data.uniform_(-initrange, initrange)
        self.out.bias.data.fill_(0)

    def forward(self, init_hidden, context=None, inputs=None, lens=None):
        batch_size, maxlen = inputs.size()
        if self.embedding is not None:
            inputs = self.embedding(inputs)
        if context is not None:
            repeated_context = context.unsqueeze(1).repeat(1, maxlen, 1)
            inputs = torch.cat([inputs, repeated_context], 2)
        inputs = F.dropout(inputs, 0.5, self.training)
        hids, h_n = self.rnn(inputs, init_hidden.unsqueeze(0))
        decoded = self.out(hids.contiguous().view(-1, self.hidden_size))  # reshape before linear over vocab
        decoded = decoded.view(batch_size, maxlen, self.vocab_size)
        return decoded

    def sampling(self, init_hidden, context, maxlen, SOS_tok, EOS_tok, mode='greedy'):
        batch_size = init_hidden.size(0)
        decoded_words = np.zeros((batch_size, maxlen), dtype=np.int)
        sample_lens = np.zeros(batch_size, dtype=np.int)

        decoder_input = gVar(torch.LongTensor([[SOS_tok] * batch_size]).view(batch_size, 1))
        decoder_input = self.embedding(decoder_input) if self.embedding is not None else decoder_input
        decoder_input = torch.cat([decoder_input, context.unsqueeze(1)], 2) if context is not None else decoder_input
        decoder_hidden = init_hidden.unsqueeze(0)
        for di in range(maxlen):
            decoder_output, decoder_hidden = self.rnn(decoder_input, decoder_hidden)
            decoder_output = self.out(decoder_output)
            if mode == 'greedy':
                topi = decoder_output[:, -1].max(1, keepdim=True)[1]
            elif mode == 'sample':
                topi = torch.multinomial(F.softmax(decoder_output[:, -1], dim=1), 1)
            decoder_input = self.embedding(topi) if self.embedding is not None else topi
            decoder_input = torch.cat([decoder_input, context.unsqueeze(1)],
                                      2) if context is not None else decoder_input
            ni = topi.squeeze().data.cpu().numpy()
            decoded_words[:, di] = ni

        for i in range(batch_size):
            for word in decoded_words[i]:
                if word == EOS_tok:
                    break
                sample_lens[i] = sample_lens[i] + 1
        return decoded_words, sample_lens


class DialogWAE(nn.Module):
    def __init__(self, config, vocab_size, PAD_token=0):
        super(DialogWAE, self).__init__()
        self.vocab_size = vocab_size
        self.maxlen = config['maxlen']
        self.clip = config['clip']
        self.lambda_gp = config['lambda_gp']
        self.temp = config['temp']

        self.embedder = nn.Embedding(vocab_size, config['emb_size'], padding_idx=PAD_token)
        self.utt_encoder = Encoder(self.embedder, config['emb_size'], config['n_hidden'],
                                   True, config['n_layers'], config['noise_radius'])
        self.context_encoder = ContextEncoder(self.utt_encoder, config['n_hidden'] * 2 + 2, config['n_hidden'], 1,
                                              config['noise_radius'])
        self.prior_net = Variation(config['n_hidden'], config['z_size'])  # p(e|c)
        self.post_net = Variation(config['n_hidden'] * 3, config['z_size'])  # q(e|c,x)

        self.post_generator = nn.Sequential(
            nn.Linear(config['z_size'], config['z_size']),
            nn.BatchNorm1d(config['z_size'], eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Linear(config['z_size'], config['z_size']),
            nn.BatchNorm1d(config['z_size'], eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Linear(config['z_size'], config['z_size'])
        )
        self.post_generator.apply(self.init_weights)

        self.prior_generator = nn.Sequential(
            nn.Linear(config['z_size'], config['z_size']),
            nn.BatchNorm1d(config['z_size'], eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Linear(config['z_size'], config['z_size']),
            nn.BatchNorm1d(config['z_size'], eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Linear(config['z_size'], config['z_size'])
        )
        self.prior_generator.apply(self.init_weights)

        self.decoder = Decoder(self.embedder, config['emb_size'], config['n_hidden'] + config['z_size'],
                               vocab_size, n_layers=1)

        self.discriminator = nn.Sequential(
            nn.Linear(config['n_hidden'] + config['z_size'], config['n_hidden'] * 2),
            nn.BatchNorm1d(config['n_hidden'] * 2, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Linear(config['n_hidden'] * 2, config['n_hidden'] * 2),
            nn.BatchNorm1d(config['n_hidden'] * 2, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Linear(config['n_hidden'] * 2, 1),
        )
        self.discriminator.apply(self.init_weights)

        self.optimizer_AE = optim.SGD(list(self.context_encoder.parameters())
                                      + list(self.post_net.parameters())
                                      + list(self.post_generator.parameters())
                                      + list(self.decoder.parameters()), lr=config['lr_ae'])
        self.optimizer_G = optim.RMSprop(list(self.post_net.parameters())
                                         + list(self.post_generator.parameters())
                                         + list(self.prior_net.parameters())
                                         + list(self.prior_generator.parameters()), lr=config['lr_gan_g'])
        self.optimizer_D = optim.RMSprop(self.discriminator.parameters(), lr=config['lr_gan_d'])

        self.lr_scheduler_AE = optim.lr_scheduler.StepLR(self.optimizer_AE, step_size=10, gamma=0.6)

        self.criterion_ce = nn.CrossEntropyLoss()

        self.one = gData(torch.FloatTensor([1]))
        self.minus_one = self.one * -1

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.uniform_(-0.02, 0.02)
            m.bias.data.fill_(0)

    def sample_code_post(self, x, c):
        e, _, _ = self.post_net(torch.cat((x, c), 1))
        z = self.post_generator(e)
        return z

    def sample_code_prior(self, c):
        e, _, _ = self.prior_net(c)
        z = self.prior_generator(e)
        return z

    def train_AE(self, context, context_lens, utt_lens, floors, response, res_lens):
        self.context_encoder.train()
        self.decoder.train()
        c = self.context_encoder(context, context_lens, utt_lens, floors)
        x, _ = self.utt_encoder(response[:, 1:], res_lens - 1)
        z = self.sample_code_post(x, c)
        output = self.decoder(torch.cat((z, c), 1), None, response[:, :-1], (res_lens - 1))
        flattened_output = output.view(-1, self.vocab_size)

        dec_target = response[:, 1:].contiguous().view(-1)
        mask = dec_target.gt(0)  # [(batch_sz*seq_len)]
        masked_target = dec_target.masked_select(mask)  #
        output_mask = mask.unsqueeze(1).expand(mask.size(0), self.vocab_size)  # [(batch_sz*seq_len) x n_tokens]
        masked_output = flattened_output.masked_select(output_mask).view(-1, self.vocab_size)

        self.optimizer_AE.zero_grad()
        loss = self.criterion_ce(masked_output / self.temp, masked_target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(list(self.context_encoder.parameters()) + list(self.decoder.parameters()),
                                       self.clip)
        self.optimizer_AE.step()

        return [('train_loss_AE', loss.item())]

    def train_G(self, context, context_lens, utt_lens, floors, response, res_lens):
        self.context_encoder.eval()
        self.optimizer_G.zero_grad()

        for p in self.discriminator.parameters():
            p.requires_grad = False

        c = self.context_encoder(context, context_lens, utt_lens, floors)
        # -----------------posterior samples ---------------------------
        x, _ = self.utt_encoder(response[:, 1:], res_lens - 1)
        z_post = self.sample_code_post(x.detach(), c.detach())
        errG_post = torch.mean(self.discriminator(torch.cat((z_post, c.detach()), 1)))
        errG_post.backward(self.minus_one)

        # ----------------- prior samples ---------------------------
        prior_z = self.sample_code_prior(c.detach())
        errG_prior = torch.mean(self.discriminator(torch.cat((prior_z, c.detach()), 1)))
        errG_prior.backward(self.one)

        self.optimizer_G.step()

        for p in self.discriminator.parameters():
            p.requires_grad = True

        costG = errG_prior - errG_post
        return [('train_loss_G', costG.item())]

    def train_D(self, context, context_lens, utt_lens, floors, response, res_lens):
        self.context_encoder.eval()
        self.discriminator.train()

        self.optimizer_D.zero_grad()

        batch_size = context.size(0)

        c = self.context_encoder(context, context_lens, utt_lens, floors)
        x, _ = self.utt_encoder(response[:, 1:], res_lens - 1)
        post_z = self.sample_code_post(x, c)
        errD_post = torch.mean(self.discriminator(torch.cat((post_z.detach(), c.detach()), 1)))
        errD_post.backward(self.one)

        prior_z = self.sample_code_prior(c)
        errD_prior = torch.mean(self.discriminator(torch.cat((prior_z.detach(), c.detach()), 1)))
        errD_prior.backward(self.minus_one)

        alpha = gData(torch.rand(batch_size, 1))
        alpha = alpha.expand(prior_z.size())
        interpolates = alpha * prior_z.data + ((1 - alpha) * post_z.data)
        interpolates = Variable(interpolates, requires_grad=True)
        d_input = torch.cat((interpolates, c.detach()), 1)
        disc_interpolates = torch.mean(self.discriminator(d_input))
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=gData(torch.ones(disc_interpolates.size())),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.contiguous().view(gradients.size(0), -1).norm(2,
                                                                                     dim=1) - 1) ** 2).mean() * self.lambda_gp
        gradient_penalty.backward()

        self.optimizer_D.step()
        costD = -(errD_prior - errD_post) + gradient_penalty
        return [('train_loss_D', costD.item())]

    def valid(self, context, context_lens, utt_lens, floors, response, res_lens):
        self.context_encoder.eval()
        self.discriminator.eval()
        self.decoder.eval()

        c = self.context_encoder(context, context_lens, utt_lens, floors)
        x, _ = self.utt_encoder(response[:, 1:], res_lens - 1)
        post_z = self.sample_code_post(x, c)
        prior_z = self.sample_code_prior(c)
        errD_post = torch.mean(self.discriminator(torch.cat((post_z, c), 1)))
        errD_prior = torch.mean(self.discriminator(torch.cat((prior_z, c), 1)))
        costD = -(errD_prior - errD_post)
        costG = -costD

        dec_target = response[:, 1:].contiguous().view(-1)
        mask = dec_target.gt(0)  # [(batch_sz*seq_len)]
        masked_target = dec_target.masked_select(mask)
        output_mask = mask.unsqueeze(1).expand(mask.size(0), self.vocab_size)
        output = self.decoder(torch.cat((post_z, c), 1), None, response[:, :-1], (res_lens - 1))
        flattened_output = output.view(-1, self.vocab_size)
        masked_output = flattened_output.masked_select(output_mask).view(-1, self.vocab_size)
        lossAE = self.criterion_ce(masked_output / self.temp, masked_target)
        return [('valid_loss_AE', lossAE.item()), ('valid_loss_G', costG.item()), ('valid_loss_D', costD.item())]

    def sample(self, context, context_lens, utt_lens, floors, repeat, SOS_tok, EOS_tok):
        self.context_encoder.eval()
        self.decoder.eval()

        c = self.context_encoder(context, context_lens, utt_lens, floors)
        c_repeated = c.expand(repeat, -1)
        prior_z = self.sample_code_prior(c_repeated)
        sample_words, sample_lens = self.decoder.sampling(torch.cat((prior_z, c_repeated), 1),
                                                          None, self.maxlen, SOS_tok, EOS_tok, "greedy")
        return sample_words, sample_lens

    def adjust_lr(self):
        self.lr_scheduler_AE.step()
