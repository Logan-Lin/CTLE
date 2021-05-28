from itertools import zip_longest

import numpy as np
import torch
from torch import nn
from sklearn.utils import shuffle
from torch.nn.utils.rnn import pack_padded_sequence

from utils import next_batch


class HierEmbedding(nn.Module):
    def __init__(self, token_embed_size, num_vocab, week_embed_size, hour_embed_size, duration_embed_size):
        super().__init__()
        self.num_vocab = num_vocab
        self.token_embed_size = token_embed_size
        self.embed_size = token_embed_size + week_embed_size + hour_embed_size + duration_embed_size

        self.token_embed = nn.Embedding(num_vocab, token_embed_size)
        self.token_embed.weight.data.uniform_(-0.5/token_embed_size, 0.5/token_embed_size)
        self.week_embed = nn.Embedding(7, week_embed_size)
        self.hour_embed = nn.Embedding(24, hour_embed_size)
        self.duration_embed = nn.Embedding(24, duration_embed_size)

        self.dropout = nn.Dropout(0.1)

    def forward(self, token, week, hour, duration):
        token = self.token_embed(token)
        week = self.week_embed(week)
        hour = self.hour_embed(hour)
        duration = self.duration_embed(duration)

        return self.dropout(torch.cat([token, week, hour, duration], dim=-1))


class Hier(nn.Module):
    def __init__(self, embed: HierEmbedding, hidden_size, num_layers, share=True, dropout=0.1):
        super().__init__()
        self.embed = embed
        self.add_module('embed', self.embed)
        self.encoder = nn.LSTM(self.embed.embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        if share:
            self.out_linear = nn.Sequential(nn.Linear(hidden_size, self.embed.token_embed_size), nn.LeakyReLU())
        else:
            self.out_linear = nn.Sequential(nn.Linear(hidden_size, self.embed.token_embed_size),
                                            nn.LeakyReLU(),
                                            nn.Linear(self.embed.token_embed_size, self.embed.num_vocab))
        self.share = share

    def forward(self, token, week, hour, duration, valid_len, **kwargs):
        """
        :param token: sequences of tokens, shape (batch, seq_len)
        :param week: sequences of week indices, shape (batch, seq_len)
        :param hour: sequences of visit time slot indices, shape (batch, seq_len)
        :param duration: sequences of duration slot indices, shape (batch, seq_len)
        :return: the output prediction of next vocab, shape (batch, seq_len, num_vocab)
        """
        embed = self.embed(token, week, hour, duration)  # (batch, seq_len, embed_size)
        packed_embed = pack_padded_sequence(embed, valid_len, batch_first=True, enforce_sorted=False)
        encoder_out, hc = self.encoder(packed_embed)  # (batch, seq_len, hidden_size)
        out = self.out_linear(encoder_out.data)  # (batch, seq_len, token_embed_size)

        if self.share:
            out = torch.matmul(out, self.embed.token_embed.weight.transpose(0, 1))  # (total_valid_len, num_vocab)
        return out

    def static_embed(self):
        return self.embed.token_embed.weight[:self.embed.num_vocab].detach().cpu().numpy()


def train_hier(dataset, hier_model, num_epoch, batch_size, device):
    user_ids, src_tokens, src_weekdays, src_ts, src_lens = zip(*dataset.gen_sequence(select_days=0))
    hier_model = hier_model.to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(hier_model.parameters())
    for epoch in range(num_epoch):
        for batch in next_batch(shuffle(list(zip(src_tokens, src_weekdays, src_ts, src_lens))), batch_size=batch_size):
            src_token, src_weekday, src_t, src_len = zip(*batch)
            src_token, src_weekday = [torch.from_numpy(np.transpose(np.array(list(zip_longest(*item, fillvalue=0))))).long().to(device)
                                      for item in (src_token, src_weekday)]
            src_t = torch.from_numpy(np.transpose(np.array(list(zip_longest(*src_t, fillvalue=0))))).float().to(device)
            src_len = torch.tensor(src_len).long().to(device)

            src_hour = (src_t % (24 * 60 * 60) / 60 / 60).long()
            src_duration = ((src_t[:, 1:] - src_t[:, :-1]) % (24 * 60 * 60) / 60 / 60).long()
            src_duration = torch.clamp(src_duration, 0, 23)

            hier_out = hier_model(token=src_token[:, :-1], week=src_weekday[:, :-1], hour=src_hour[:, :-1],
                                   duration=src_duration, valid_len=src_len-1)  # (batch, seq_len, num_vocab)
            trg_token = pack_padded_sequence(src_token[:, 1:], src_len-1, batch_first=True, enforce_sorted=False).data
            loss = loss_func(hier_out, trg_token)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return hier_model.static_embed()
