
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Encoder(nn.Module):

    def __init__(self, vocab_size=10002,
                 enc_word_dim=256,
                 enc_h_dim=256,
                 enc_num_layers=1,
                 latent_dim=32):
        super(Encoder, self).__init__()

        self.enc_h_dim = enc_h_dim
        self.enc_num_layers = enc_num_layers
        self.enc_word_dim = enc_word_dim

        self.enc_word_vecs = nn.Embedding(vocab_size, self.enc_word_dim)
        self.latent_linear = nn.Linear(enc_h_dim + latent_dim, latent_dim)
        self.enc_rnn = nn.LSTM(enc_word_dim, enc_h_dim, num_layers=enc_num_layers, batch_first=True)

    def forward(self, sents, eps):
        word_vecs = self.enc_word_vecs(sents)
        h0 = torch.zeros((self.enc_num_layers, word_vecs.size(0), self.enc_h_dim), device=sents.device)
        c0 = torch.zeros((self.enc_num_layers, word_vecs.size(0), self.enc_h_dim), device=sents.device)

        enc_h_states, _ = self.enc_rnn(word_vecs, (h0, c0))
        enc_h_states_last = enc_h_states[:, -1]

        z = self.latent_linear(torch.cat((enc_h_states_last, eps), dim=1))
        return z, enc_h_states_last.data


class Encoder_AE(nn.Module):

    def __init__(self, vocab_size=10002,
                 enc_word_dim=256,
                 enc_h_dim=256,
                 enc_num_layers=1,
                 latent_dim=32):
        super(Encoder_AE, self).__init__()

        self.enc_h_dim = enc_h_dim
        self.enc_num_layers = enc_num_layers
        self.enc_word_dim = enc_word_dim

        self.enc_word_vecs = nn.Embedding(vocab_size, self.enc_word_dim)
        self.latent_linear = nn.Linear(enc_h_dim, latent_dim)
        self.enc_rnn = nn.LSTM(enc_word_dim, enc_h_dim, num_layers=enc_num_layers, batch_first=True)

    def forward(self, sents):
        word_vecs = self.enc_word_vecs(sents)
        h0 = torch.zeros((self.enc_num_layers, word_vecs.size(0), self.enc_h_dim), device=sents.device)
        c0 = torch.zeros((self.enc_num_layers, word_vecs.size(0), self.enc_h_dim), device=sents.device)

        enc_h_states, _ = self.enc_rnn(word_vecs, (h0, c0))
        enc_h_states_last = enc_h_states[:, -1]

        z = self.latent_linear(enc_h_states_last)
        return z


class Nu_xz(nn.Module):

    def __init__(self, enc_h_dim=256, latent_dim=32):
        super(Nu_xz, self).__init__()

        self.Nu_fc1 = nn.Linear(latent_dim + enc_h_dim, 512)
        self.Nu_fc2 = nn.Linear(512, 512)
        self.Nu_fc3 = nn.Linear(512, 512)
        self.Nu_fc4 = nn.Linear(512, 1)

    def forward(self, z, enc):
        z = torch.cat((z, enc), dim=1)

        z = F.leaky_relu(self.Nu_fc1(z))
        z = F.leaky_relu(self.Nu_fc2(z))
        z = F.leaky_relu(self.Nu_fc3(z))
        z = self.Nu_fc4(z)  # log nu
        return z


class Nu_z(nn.Module):

    def __init__(self, latent_dim=32):
        super(Nu_z, self).__init__()

        self.Nu_fc1 = nn.Linear(latent_dim, 512)
        self.Nu_fc2 = nn.Linear(512, 512)
        self.Nu_fc3 = nn.Linear(512, 512)
        self.Nu_fc4 = nn.Linear(512, 1)

    def forward(self, z):
        z = F.leaky_relu(self.Nu_fc1(z))
        z = F.leaky_relu(self.Nu_fc2(z))
        z = F.leaky_relu(self.Nu_fc3(z))
        z = self.Nu_fc4(z)  # log nu
        return z


class Decoder(nn.Module):

    def __init__(self,vocab_size=10002,
                 dec_word_dim=256,
                 dec_h_dim=256,
                 dec_num_layers=1,
                 dec_dropout=0.5,
                 latent_dim=32):
        super(Decoder, self).__init__()

        self.dec_h_dim = dec_h_dim
        self.dec_num_layers = dec_num_layers
        self.dec_word_dim = dec_word_dim

        self.dec_word_vecs = nn.Embedding(vocab_size, self.dec_word_dim)
        dec_input_size = dec_word_dim + latent_dim
        self.dec_rnn = nn.LSTM(dec_input_size, dec_h_dim, num_layers=dec_num_layers, batch_first=True)
        self.dec_linear = nn.Sequential(*[nn.Dropout(dec_dropout), nn.Linear(dec_h_dim, vocab_size), nn.LogSoftmax(dim=1)])

        self.dropout = nn.Dropout(dec_dropout)
        self.latent_hidden_linear = nn.Linear(latent_dim, dec_h_dim)

    def forward(self, sents, q_z):
        self.word_vecs = self.dropout(self.dec_word_vecs(sents[:, :-1]))
        self.h0 = torch.zeros((self.dec_num_layers, self.word_vecs.size(0), self.dec_h_dim), device=sents.device)
        self.c0 = torch.zeros((self.dec_num_layers, self.word_vecs.size(0), self.dec_h_dim), device=sents.device)
        self.h0[-1] = self.latent_hidden_linear(q_z)

        q_z_expand = q_z.unsqueeze(1).expand(self.word_vecs.size(0), self.word_vecs.size(1), q_z.size(1))
        dec_input = torch.cat([self.word_vecs, q_z_expand], 2)

        memory, _ = self.dec_rnn(dec_input, (self.h0, self.c0))
        dec_linear_input = memory.contiguous()

        preds = self.dec_linear(dec_linear_input.view(self.word_vecs.size(0) * self.word_vecs.size(1), -1)).\
            view(self.word_vecs.size(0), self.word_vecs.size(1), -1)
        return preds


class RNNVAE(nn.Module):
    def __init__(self, vocab_size=10000,
                 enc_word_dim=256,
                 enc_h_dim=256,
                 enc_num_layers=1,
                 dec_word_dim=256,
                 dec_h_dim=256,
                 dec_num_layers=1,
                 dec_dropout=0.5,
                 latent_dim=32,
                 mode='savae'):
        super(RNNVAE, self).__init__()
        self.enc_h_dim = enc_h_dim
        self.enc_num_layers = enc_num_layers
        self.dec_h_dim = dec_h_dim
        self.dec_num_layers = dec_num_layers

        self.enc_word_vecs = nn.Embedding(vocab_size, enc_word_dim)
        self.latent_linear_mean = nn.Linear(enc_h_dim, latent_dim)
        self.latent_linear_logvar = nn.Linear(enc_h_dim, latent_dim)
        self.enc_rnn = nn.LSTM(enc_word_dim, enc_h_dim, num_layers=enc_num_layers,
                               batch_first=True)
        self.enc = nn.ModuleList([self.enc_word_vecs, self.enc_rnn,
                                  self.latent_linear_mean, self.latent_linear_logvar])

        self.dec_word_vecs = nn.Embedding(vocab_size, dec_word_dim)
        dec_input_size = dec_word_dim
        dec_input_size += latent_dim
        self.dec_rnn = nn.LSTM(dec_input_size, dec_h_dim, num_layers=dec_num_layers,
                               batch_first=True)
        self.dec_linear = nn.Sequential(*[nn.Dropout(dec_dropout),
                                          nn.Linear(dec_h_dim, vocab_size),
                                          nn.LogSoftmax(dim=1)])
        self.dropout = nn.Dropout(dec_dropout)
        self.dec = nn.ModuleList([self.dec_word_vecs, self.dec_rnn, self.dec_linear])
        if latent_dim > 0:
            self.latent_hidden_linear = nn.Linear(latent_dim, dec_h_dim)
            self.dec.append(self.latent_hidden_linear)

    def _enc_forward(self, sent):
        word_vecs = self.enc_word_vecs(sent)
        h0 = Variable(torch.zeros(self.enc_num_layers, word_vecs.size(0),
                                  self.enc_h_dim).type_as(word_vecs.data))
        c0 = Variable(torch.zeros(self.enc_num_layers, word_vecs.size(0),
                                  self.enc_h_dim).type_as(word_vecs.data))
        enc_h_states, _ = self.enc_rnn(word_vecs, (h0, c0))
        enc_h_states_last = enc_h_states[:, -1]
        mean = self.latent_linear_mean(enc_h_states_last)
        logvar = self.latent_linear_logvar(enc_h_states_last)
        return mean, logvar

    def _reparameterize(self, mean, logvar, z=None):
        std = logvar.mul(0.5).exp()
        if z is None:
            z = Variable(torch.cuda.FloatTensor(std.size()).normal_(0, 1))
        return z.mul(std) + mean

    def _dec_forward(self, sent, q_z, init_h=True):
        self.word_vecs = self.dropout(self.dec_word_vecs(sent[:, :-1]))
        if init_h:
            self.h0 = Variable(torch.zeros(self.dec_num_layers, self.word_vecs.size(0),
                                           self.dec_h_dim).type_as(self.word_vecs.data), requires_grad=False)
            self.c0 = Variable(torch.zeros(self.dec_num_layers, self.word_vecs.size(0),
                                           self.dec_h_dim).type_as(self.word_vecs.data), requires_grad=False)
        else:
            self.h0.data.zero_()
            self.c0.data.zero_()

        if q_z is not None:
            q_z_expand = q_z.unsqueeze(1).expand(self.word_vecs.size(0),
                                                 self.word_vecs.size(1), q_z.size(1))
            dec_input = torch.cat([self.word_vecs, q_z_expand], 2)
        else:
            dec_input = self.word_vecs
        if q_z is not None:
            self.h0[-1] = self.latent_hidden_linear(q_z)
        memory, _ = self.dec_rnn(dec_input, (self.h0, self.c0))
        dec_linear_input = memory.contiguous()
        preds = self.dec_linear(dec_linear_input.view(
            self.word_vecs.size(0) * self.word_vecs.size(1), -1)).view(
            self.word_vecs.size(0), self.word_vecs.size(1), -1)
        return preds
