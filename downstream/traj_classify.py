import numpy as np
import torch
from sklearn.utils import shuffle
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.metrics import accuracy_score, recall_score

from utils import weight_init, create_src, next_batch


class FCTrajectoryClassifier(nn.Module):
    def __init__(self, pooling_type, input_size, hidden_size, output_size):
        """
        @param pooling_type: type of pooling method, choose from 'max', 'mean' and 'lstm'.
        @param input_size: size of input vectors.
        @param hidden_size: hidden size of FC classifier.
        @param output_size: number of classes to classify.
        """
        super().__init__()
        self.lstm_pooling = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)

        self.pooling_type = pooling_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.pool_mlp = nn.Linear(input_size, hidden_size)
        self.pre_mlp = nn.Sequential(nn.Tanh(), nn.LeakyReLU(), nn.Linear(hidden_size, int(hidden_size/4)),
                                     nn.LeakyReLU(), nn.Linear(int(hidden_size/4), output_size))
        self.apply(weight_init)

    def forward(self, embed_seq, valid_len):
        """
        @param embed_seq: input embedding sequence of trajectories, shape (batch_size, seq_len, input_size).
        @param valid_len: valid length of each sequence in this batch, shape (batch_size)
        """
        batch_size = embed_seq.size(0)
        seq_len = embed_seq.size(1)

        if self.pooling_type == 'lstm':
            packed_embed_seq = pack_padded_sequence(embed_seq, valid_len, batch_first=True, enforce_sorted=False)
            _, (h, c) = self.lstm_pooling(packed_embed_seq)
            pooled_embed = h.squeeze(0)  # (batch_size, hidden_size)
        else:
            embed_seq = self.pool_mlp(embed_seq)  # (batch_size, seq_len, hidden_size)

            mask = torch.zeros(batch_size, seq_len).bool().to(embed_seq.device)
            for i, l in enumerate(valid_len):
                mask[i, l:] = True
            embed_seq = embed_seq.masked_fill(mask.unsqueeze(-1), float('-inf') if self.pooling_type == 'max' else 0.0)

            pooled_embed = embed_seq.max(dim=1)[0] if self.pooling_type == 'max' \
                else embed_seq.sum(dim=1) / valid_len.unsqueeze(1)  # (batch_size, hidden_size)
        pre = self.pre_mlp(pooled_embed)  # (batch_size, output_size)
        return pre


def fc_trajectory_classify(dataset, embed_model, pre_model, num_epoch, batch_size, device):
    pre_model = pre_model.to(device)
    embed_model = embed_model.to(device)
    optimizer = torch.optim.Adam(list(pre_model.parameters()) + list(embed_model.parameters()), lr=1e-4)
    loss_func = nn.CrossEntropyLoss()

    train_set = dataset.gen_sequence(select_days=0)
    test_set = dataset.gen_sequence(select_days=1)

    def pre_func(batch):
        user_index, full_seq, weekday, timestamp, length = zip(*batch)

        weekday, length = (torch.tensor(item).long().to(device) for item in [weekday, length])
        weekend = (weekday == 5) + (weekday == 6)
        full_seq = torch.from_numpy(create_src(full_seq, fill_value=dataset.num_loc)).long().to(device)
        timestamp = torch.from_numpy(create_src(timestamp, fill_value=0)).float().to(device)

        embed_seq = embed_model(full_seq, timestamp=timestamp)  # (batch_size, seq_len, input_size)
        pre = pre_model(embed_seq, length)
        return pre, weekend.long()

    score_log = []
    test_point = int(len(train_set) / batch_size / 6) - 1
    for epoch in range(num_epoch):
        for i, batch in enumerate(next_batch(shuffle(train_set), batch_size)):
            out, label = pre_func(batch)
            loss = loss_func(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % test_point == 0:
                pres, labels = [], []
                for test_batch in next_batch(test_set, batch_size):
                    test_out, test_label = pre_func(test_batch)
                    pres.append(test_out.argmax(-1).detach().cpu().numpy())
                    labels.append(test_label.detach().cpu().numpy())
                pres, labels = np.concatenate(pres), np.concatenate(labels)
                acc, recall = accuracy_score(labels, pres), recall_score(labels, pres, average='macro')
                score_log.append([acc, recall])

    best_acc, best_recall = np.max(score_log, axis=0)
    print('Acc %.6f, Recall %.6f' % (best_acc, best_recall))
