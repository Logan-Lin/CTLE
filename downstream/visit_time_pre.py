import math
from abc import ABC

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, recall_score
from sklearn.utils import shuffle
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn import functional as F
from scipy import integrate

from utils import next_batch, create_src_trg, weight_init, create_src, mean_absolute_percentage_error


class ERPPTimePredictor(nn.Module, ABC):
    def __init__(self, embed_layer, input_size, lstm_hidden_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = lstm_hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embed_layer = embed_layer
        self.add_module('embed_layer', self.embed_layer)
        self.encoder = nn.LSTM(input_size + 1, lstm_hidden_size, num_layers, dropout=0.1, batch_first=True)
        self.mlp = nn.Sequential(nn.Tanh(), nn.Linear(lstm_hidden_size, hidden_size))
        self.event_linear = nn.Linear(hidden_size, output_size)
        self.time_linear = nn.Linear(hidden_size, 1)

    def forward(self, input_time, input_events, valid_len, pre_len, **kwargs):
        event_embedding = self.embed_layer(input_events, downstream=True, pre_len=pre_len,
                                           **kwargs)  # (batch, seq_len, input_size)
        lstm_input = torch.cat([event_embedding, input_time.unsqueeze(-1)], dim=-1)  # (batch, seq_len, input_size + 1)
        max_len = valid_len.max()
        lstm_input = torch.stack([
            torch.cat([lstm_input[i, :s - pre_len], lstm_input[i, -pre_len:-1],
                       torch.zeros(max_len - s, self.input_size + 1).float().to(input_time.device)])
            for i, s in enumerate(valid_len)
        ])

        lstm_out, _ = self.encoder(lstm_input)  # (batch, seq_len, lstm_hidden_size)
        lstm_out_pre = torch.stack([lstm_out[i, s - pre_len:s - 1] for i, s in
                                    enumerate(valid_len)])  # (batch_size, pre_len, lstm_hidden_size)

        mlp_out = self.mlp(lstm_out_pre)  # (batch_size, pre_len, hidden_size)
        event_out = self.event_linear(mlp_out)  # (batch_size, pre_len, num_events)
        time_out = self.time_linear(mlp_out)  # (batch_size, pre_len, 1)
        return event_out, time_out


class RMTPPTimePredictor(ERPPTimePredictor, ABC):
    def __init__(self, embed_layer, input_size, lstm_hidden_size, hidden_size, output_size, num_layers):
        super().__init__(embed_layer, input_size, lstm_hidden_size, hidden_size, output_size, num_layers)
        self.intensity_w = nn.Parameter(torch.tensor(0.1).float(), requires_grad=True)
        self.intensity_b = nn.Parameter(torch.tensor(0.1).float(), requires_grad=True)

    def rmtpp_loss(self, hidden_things, time_duration):
        loss = torch.mean(hidden_things + self.intensity_w * time_duration + self.intensity_b +
                          (torch.exp(hidden_things + self.intensity_b) -
                           torch.exp(hidden_things + self.intensity_w * time_duration + self.intensity_b)) / self.intensity_w)
        return -loss

    def duration_pre(self, hidden_things):
        def _equ(time_var, time_cif, w, b):
            return time_var * np.exp(time_cif + w * time_var + b +
                                     (np.exp(time_cif + b) -
                                      np.exp(time_cif + w * time_var + b)) / w)

        func = lambda x: _equ(x, hidden_things, self.intensity_w.item(), self.intensity_b.item())
        duration = integrate.quad(func, 0, np.inf)[0]
        return duration


def erpp_visit_time_prediction(dataset, pre_model, pre_len, num_epoch, batch_size, device, use_event_loss):
    pre_model = pre_model.to(device)
    optimizer = torch.optim.Adam(pre_model.parameters(), lr=1e-4)
    event_loss_func = nn.CrossEntropyLoss()
    time_loss_func = nn.MSELoss()

    train_set = dataset.gen_sequence(min_len=pre_len + 1, select_days=0)
    test_set = dataset.gen_sequence(min_len=pre_len + 1, select_days=1)

    def pre_func(batch):
        user_index, full_seq, weekday, timestamp, length = zip(*batch)
        user_index, length = (torch.tensor(item).long().to(device) for item in (user_index, length))

        src_event, trg_event = create_src_trg(full_seq, pre_len, fill_value=dataset.num_loc)
        src_event, trg_event = (torch.from_numpy(item).long().to(device) for item in [src_event, trg_event])
        full_event = torch.cat([src_event, trg_event], dim=-1)

        if isinstance(pre_model, RMTPPTimePredictor):
            timestamp = [np.diff(t_row, prepend=t_row[0]).tolist() for t_row in timestamp]
        src_t, trg_t = create_src_trg(timestamp, pre_len, fill_value=0)
        src_t, trg_t = (torch.from_numpy(item).float().to(device) for item in [src_t, trg_t])
        full_t = torch.cat([src_t, trg_t], dim=-1)
        src_time, trg_time = (item % (24 * 60 * 60) / 60 / 60 for item in [src_t, trg_t])
        full_time = torch.cat([src_time, trg_time], dim=-1)

        event_pre, time_pre = pre_model(full_time, full_event, length, pre_len,
                                        user_index=user_index, timestamp=full_t)
        event_pre = event_pre.reshape(-1, pre_model.output_size)
        time_pre = time_pre.reshape(-1)
        event_label = trg_event[:, -(pre_len - 1):].reshape(-1)
        time_label = trg_time[:, -(pre_len - 1):].reshape(-1)
        return event_pre, time_pre, event_label, time_label

    event_log = []
    time_log = []
    test_point = int(len(train_set) / batch_size / 2)
    if isinstance(pre_model, RMTPPTimePredictor):
        test_point = int(len(train_set) / batch_size) - 1
    for epoch in range(num_epoch):
        for i, batch in enumerate(next_batch(shuffle(train_set), batch_size)):
            event_pre, time_pre, event_label, time_label = pre_func(batch)
            event_loss = event_loss_func(event_pre, event_label)
            if isinstance(pre_model, RMTPPTimePredictor):
                time_loss = pre_model.rmtpp_loss(time_pre, time_label)
            else:
                time_loss = time_loss_func(time_pre, time_label)
            loss = time_loss
            if use_event_loss:
                loss += event_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % test_point == 0:
                event_pres, event_labels, time_pres, time_labels = [], [], [], []
                for test_batch in next_batch(test_set, batch_size):
                    event_pre, time_pre, event_label, time_label = pre_func(test_batch)
                    event_pres.append(event_pre.argmax(-1).detach().cpu().numpy())
                    event_labels.append(event_label.detach().cpu().numpy())
                    time_pres.append(time_pre.detach().cpu().numpy())
                    time_labels.append(time_label.detach().cpu().numpy())
                event_pres, event_labels, time_pres, time_labels = (np.concatenate(item)
                                                                    for item in
                                                                    [event_pres, event_labels, time_pres, time_labels])
                if isinstance(pre_model, RMTPPTimePredictor):
                    time_pres = np.array([pre_model.duration_pre(time_pre) for time_pre in time_pres])
                acc, recall = accuracy_score(event_labels, event_pres), \
                              recall_score(event_labels, event_pres, average='macro')
                mae, rmse = mean_absolute_error(time_labels, time_pres), \
                            math.sqrt(mean_squared_error(time_labels, time_pres))
                event_log.append([acc, recall])
                time_log.append([mae, rmse])

    best_acc, best_rec = np.max(event_log, axis=0)
    best_mae, best_rmse = np.min(time_log, axis=0)
    print('Acc %.6f, Recall %.6f, MAE %.6f, RMSE %.6f' % (best_acc, best_rec, best_mae, best_rmse), flush=True)


class LSTMTimePredictor(nn.Module):
    def __init__(self, embed_layer, input_size, lstm_hidden_size, fc_hidden_size, output_size, num_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = lstm_hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embed_layer = embed_layer
        self.add_module('embed_layer', self.embed_layer)
        self.time_encoder = nn.LSTM(1+input_size, lstm_hidden_size, num_layers, dropout=0.1, batch_first=True)
        self.pre_linear = nn.Sequential(nn.Linear(lstm_hidden_size + input_size, fc_hidden_size), nn.LeakyReLU(),
                                        nn.Linear(fc_hidden_size, int(fc_hidden_size / 4)), nn.LeakyReLU(),
                                        nn.Linear(int(fc_hidden_size / 4), 1), nn.LeakyReLU())
        self.sos_token = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.apply(weight_init)

    def forward(self, input_time, input_events, valid_len, **kwargs):
        """
        @param input_time: input visit time sequence, shape (batch_size, seq_len).
            Each row is formed by [t_1, t_2, ..., t_n, 0, 0], where n=valid_len[batch_index], 0 is a placeholder.
        @param input_events: input event sequence, shape (batch_size, seq_len).
        @param valid_len: valid length of each batch, shape (batch_size)
        """
        batch_size = input_time.size(0)

        event_embedding = self.embed_layer(input_events, **kwargs)  # (batch_size, seq_len, input_size)
        shifted_event_embedding = torch.cat([self.sos_token.reshape(1, 1, -1).repeat(batch_size, 1, 1), event_embedding], dim=1)  # (batch, seq_len+1, input_size)
        shifted_time = torch.cat([torch.zeros(batch_size, 1).float().to(input_time.device), input_time],
                                 dim=-1).unsqueeze(-1)  # (batch_size, 1 + seq_len, 1)
        lstm_input = torch.cat([shifted_event_embedding, shifted_time], dim=-1)  # (batch, seq_len+1, input_size+1)

        packed_lstm_input = pack_padded_sequence(lstm_input, valid_len, batch_first=True, enforce_sorted=False)
        time_encoder_out, _ = self.time_encoder(packed_lstm_input)
        time_encoder_out = time_encoder_out.data  # (total_valid_len, lstm_hidden_size)
        # Only use embedding vectors of locations to predict.
        event_embedding = pack_padded_sequence(event_embedding, valid_len, batch_first=True, enforce_sorted=False).data  # (total_valid_len, input_size)
        cat_hidden = torch.cat([event_embedding, time_encoder_out], dim=-1) # (total_v, input_size + lstm_hidden_size)
        time_pre = self.pre_linear(cat_hidden).squeeze(-1)  # (total_v)
        return time_pre


def lstm_visit_time_prediction(dataset, pre_model, num_epoch, batch_size, device):
    pre_model = pre_model.to(device)
    optimizer = torch.optim.Adam(pre_model.parameters(), lr=1e-4)
    time_loss = nn.MSELoss()

    train_set = dataset.gen_sequence(select_days=0)
    test_set = dataset.gen_sequence(select_days=1)

    def pre_func(batch):
        user_index, full_seq, weekday, timestamp, length = zip(*batch)
        user_index, length = (torch.tensor(item).long().to(device) for item in (user_index, length))

        full_event = torch.from_numpy(create_src(full_seq, fill_value=dataset.num_loc)).long().to(device)
        timestamp = torch.from_numpy(create_src(timestamp, fill_value=0)).float().to(device)
        # Cast hour indices to [0, 1)
        hours = timestamp % (24 * 60 * 60) / 60 / 60 / 24

        pre = pre_model(input_time=hours, input_events=full_event, valid_len=length, timestamp=timestamp)
        pre = pre.reshape(-1)
        # label = torch.stack([hours[i, s-pre_len:s] for i, s in enumerate(length)]).reshape(-1)
        label = pack_padded_sequence(hours, length, batch_first=True, enforce_sorted=False).data
        return pre, label

    score_log = []
    test_point = int(len(train_set) / batch_size / 2)
    for epoch in range(num_epoch):
        for i, batch in enumerate(next_batch(shuffle(train_set), batch_size)):
            pre, label = pre_func(batch)
            loss = time_loss(pre, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % test_point == 0:
                pres, labels = [], []
                for test_batch in next_batch(test_set, batch_size):
                    pre, label = pre_func(test_batch)
                    pres.append(pre.detach().cpu().numpy())
                    labels.append(label.detach().cpu().numpy())
                pres, labels = (np.concatenate(item) * 24 for item in [pres, labels])
                mae, mape, rmse = mean_absolute_error(labels, pres), \
                                  mean_absolute_percentage_error(labels, pres), \
                                  math.sqrt(mean_squared_error(labels, pres))
                score_log.append([mae, mape, rmse])

    best_mae, best_mape, best_rmse = np.min(score_log, axis=0)
    print('MAE %.6f, MAPE %.6f, RMSE %.6f' % (best_mae, best_mape, best_rmse), flush=True)


class ScatterVisitTimePredictor(nn.Module):
    def __init__(self, embed_layer, num_time_slots, input_size, lstm_hidden_size, fc_hidden_size, output_size, num_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = lstm_hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.num_time_slots = num_time_slots

        self.embed_layer = embed_layer
        self.add_module('embed_layer', self.embed_layer)
        self.time_embed = nn.Embedding(num_time_slots, input_size)
        self.time_encoder = nn.LSTM(input_size * 2, lstm_hidden_size, num_layers, dropout=0.1, batch_first=True)
        self.pre_linear = nn.Sequential(nn.Linear(lstm_hidden_size + input_size, fc_hidden_size), nn.LeakyReLU(),
                                        nn.Linear(fc_hidden_size, int(fc_hidden_size / 4)), nn.LeakyReLU(),
                                        nn.Linear(int(fc_hidden_size / 4), output_size))
        self.sos_token = nn.Parameter(torch.zeros(input_size * 2), requires_grad=True)
        self.apply(weight_init)

    def forward(self, input_time, input_events, valid_len, **kwargs):
        """
        @param input_time: input visit time sequence, shape (batch_size, seq_len).
            Each row is formed by [t_1, t_2, ..., t_n, 0, 0], where n=valid_len[batch_index], 0 is a placeholder.
        @param input_events: input event sequence, shape (batch_size, seq_len).
        @param valid_len: valid length of each batch, shape (batch_size)
        """
        batch_size = input_time.size(0)

        event_embedding = self.embed_layer(input_events, **kwargs)  # (batch_size, seq_len, input_size)
        time_embedding = self.time_embed(torch.floor(input_time * self.num_time_slots).long())  # (batch_size, seq_len, input_size)
        lstm_input = torch.cat([self.sos_token.reshape(1, 1, -1).repeat(batch_size, 1, 1),
                                torch.cat([event_embedding, time_embedding], dim=-1)], dim=1)  # (batch_size, seq_len+1, input_size * 2
        packed_lstm_input = pack_padded_sequence(lstm_input, valid_len, batch_first=True, enforce_sorted=False)
        time_encoder_out, _ = self.time_encoder(packed_lstm_input)
        time_encoder_out = time_encoder_out.data  # (total_valid_len, lstm_hidden_size)

        # Only use embedding vectors of locations to predict.
        packed_event_embedding = pack_padded_sequence(event_embedding, valid_len, batch_first=True, enforce_sorted=False).data  # (total_valid_len, input_size)
        cat_hidden = torch.cat([packed_event_embedding, time_encoder_out], dim=-1) # (total_v, input_size + lstm_hidden_size)
        time_pre = self.pre_linear(cat_hidden)  # (batch_size, output_size)
        return time_pre


def scatter_visit_time_prediction(dataset, pre_model, time_output_type, num_epoch, batch_size, device):
    pre_model = pre_model.to(device)
    num_time_slots = pre_model.num_time_slots
    optimizer = torch.optim.Adam(pre_model.parameters(), lr=1e-4)
    time_loss = nn.MSELoss()

    train_set = dataset.gen_sequence(select_days=0)
    test_set = dataset.gen_sequence(select_days=1)

    def pre_func(batch):
        user_index, full_seq, weekday, timestamp, length = zip(*batch)
        user_index, length = (torch.tensor(item).long().to(device) for item in (user_index, length))

        full_event = torch.from_numpy(create_src(full_seq, fill_value=dataset.num_loc)).long().to(device)
        timestamp = torch.from_numpy(create_src(timestamp, fill_value=0)).float().to(device)
        # Cast hour indices to [0, 1)
        hours = timestamp % (24 * 60 * 60) / 60 / 60 / 24

        pre = pre_model(input_time=hours, input_events=full_event, valid_len=length, timestamp=timestamp)  # (batch, output_size)
        label = pack_padded_sequence(hours, length, batch_first=True, enforce_sorted=False).data

        if time_output_type == 'scalar':
            pre_cont = pre.reshape(-1)
            mse_loss = time_loss(pre_cont, label)
        else:
            pre_slots = pre.argmax(dim=-1)  # (batch)
            if time_output_type == 'argmax':
                pre_fill_mask = pre_slots.unsqueeze(-1) != torch.arange(0, num_time_slots).long().to(device).unsqueeze(0).repeat(pre_slots.size(0), 1)
                pre = pre.masked_fill(pre_fill_mask, float('-inf'))

            pre_softmax = F.softmax(pre, dim=-1)  # (batch, num_time_slots)
            pre_cont = ((torch.arange(0.5, num_time_slots, 1).to(device) / num_time_slots).unsqueeze(0) * pre_softmax).sum(-1)
            label_slots = torch.floor(label * num_time_slots).long()
            select_mask = (pre_slots != label_slots)
            mse_loss = time_loss(pre_cont.masked_select(select_mask), label.masked_select(select_mask))  # (batch)

        return pre_cont, label, mse_loss

    score_log = []
    test_point = int(len(train_set) / batch_size / 2) - 1
    for epoch in range(num_epoch):
        for i, batch in enumerate(next_batch(shuffle(train_set), batch_size)):
            _, _, loss = pre_func(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % test_point == 0:
                pres, labels = [], []
                for test_batch in next_batch(test_set, batch_size):
                    pre, label, _ = pre_func(test_batch)
                    pres.append(pre.detach().cpu().numpy())
                    labels.append(label.detach().cpu().numpy())
                pres, labels = (np.concatenate(item) * 24 for item in [pres, labels])
                mae, mape, rmse = mean_absolute_error(labels, pres), \
                                  mean_absolute_percentage_error(labels, pres), \
                                  math.sqrt(mean_squared_error(labels, pres))
                score_log.append([mae, mape, rmse])

    best_mae, best_mape, best_rmse = np.min(score_log, axis=0)
    print('MAE %.6f, MAPE %.6f, RMSE %.6f' % (best_mae, best_mape, best_rmse), flush=True)

    import os
    torch.save(pre_model.state_dict(), os.path.join('data', 'model', 'downstream.model'))
