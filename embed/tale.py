from embed.w2v import *


def gen_all_slots(minute, time_slice_length, influence_span_length):
    """
    @param minute: UTC timestamp in minute.
    @param time_slice_length: length of one slot in seconds.
    @param influence_span_length: length of influence span in seconds.
    """
    def _cal_slice(x):
        return int((x % (24 * 60)) / time_slice_length)

    # max_num_slots = math.ceil(time_slice_length / influence_span_length) + 1
    if influence_span_length == 0:
        slices, props = [_cal_slice(minute)], [1.0]

    else:
        minute_floors = list({minute - influence_span_length / 2, minute + influence_span_length / 2} |
                             set(range((int((minute - influence_span_length/2) / time_slice_length) + 1) * time_slice_length,
                                       int(minute + influence_span_length / 2), time_slice_length)))
        minute_floors.sort()

        slices = [_cal_slice(time_minute) for time_minute in minute_floors[:-1]]
        props = [(minute_floors[index + 1] - minute_floors[index]) / influence_span_length
                 for index in range(len(minute_floors) - 1)]

        # mask_length = max_num_slots - len(slices)
        # slices += [slices[-1]] * mask_length
        # props += [0.0] * mask_length

    return slices, props


class TaleData(W2VData):
    def __init__(self, sentences, timestamps, time_slice_len, influence_span_length, indi_context=True):
        """
        @param sentences: sequences of locations.
        @param minutes: UTC minutes corresponding to sentences.
        @param time_slice_len: length of one time slice, in minute.
        """
        temp_sentence = []
        slices, props = [], []
        for sentence, timestamp in zip(sentences, timestamps):
            slice_row, prop_row = [], []
            minute = list(map(lambda x: x / 60, timestamp))
            for token, one_minute in zip(sentence, minute):
                slice, prop = gen_all_slots(one_minute, time_slice_len, influence_span_length)
                temp_sentence += ['{}-{}'.format(token, s) for s in slice]
                slice_row.append(slice)
                prop_row.append(prop)
            slices.append(slice_row)
            props.append(prop_row)

        super().__init__([temp_sentence])

        self.id2index = {id: index for index, id in enumerate(self.word_freq[:, 0])}
        self.word_freq[:, 0] = np.array([self.id2index[x] for x in self.word_freq[:, 0]])
        self.word_freq = self.word_freq.astype(int)
        self.huffman_tree = HuffmanTree(self.word_freq)

        self.sentences = sentences
        self.slices = slices
        self.props = props
        self.indi_context = indi_context

    def get_path_pairs(self, window_size):
        path_pairs = []
        for sentence, slice, prop in zip(self.sentences, self.slices, self.props):
            for i in range(0, len(sentence) - (2 * window_size + 1)):
                temp_targets = ['{}-{}'.format(sentence[i+window_size], s) for s in slice[i+window_size]]
                target_indices = [self.id2index[t] for t in temp_targets]  # (num_overlapping_slices)
                pos_paths = [self.huffman_tree.id2pos[t] for t in target_indices]
                neg_paths = [self.huffman_tree.id2neg[t] for t in target_indices]
                context = sentence[i:i+window_size] + sentence[i+window_size+1:i+2*window_size+1]
                if self.indi_context:
                    path_pairs += [[[c], pos_paths, neg_paths, prop[i+window_size]] for c in context]
                else:
                    path_pairs.append([context, pos_paths, neg_paths, prop[i+window_size]])
        return path_pairs


class Tale(HS):
    def __init__(self, num_vocab, num_temp_vocab, embed_dimension):
        super().__init__(num_vocab, embed_dimension)
        self.w_embeddings = nn.Embedding(num_temp_vocab, embed_dimension, padding_idx=0, sparse=True)

    def forward(self, pos_u, pos_w, neg_w, **kwargs):
        """
        @param pos_u: positive input tokens, shape (batch_size, window_size * 2)
        @param pos_w: positive output tokens, shape (batch_size, pos_path_len)
        @param neg_w: negative output tokens, shape (batch_size, neg_path_len)
        """
        pos_score, neg_score = super().forward(pos_u, pos_w, neg_w, sum=False)  # (batch_size, pos_path_len)
        prop = kwargs['prop']
        pos_score, neg_score = (-1 * (item.sum(axis=1) * prop).sum() for item in (pos_score, neg_score))
        return pos_score + neg_score


def train_tale(tale_model: Tale, dataset: TaleData, window_size, batch_size, num_epoch, init_lr, device):
    tale_model = tale_model.to(device)
    optimizer = torch.optim.SGD(tale_model.parameters(), lr=init_lr)

    train_set = dataset.get_path_pairs(window_size)
    trained_batches = 0
    batch_count = math.ceil(num_epoch * len(train_set) / batch_size)

    avg_loss = 0.
    for epoch in range(num_epoch):
        for pair_batch in next_batch(shuffle(train_set), batch_size):
            flatten_batch = []
            for row in pair_batch:
                flatten_batch += [[row[0], p, n, pr] for p, n, pr in zip(*row[1:])]

            context, pos_pairs, neg_pairs, prop = zip(*flatten_batch)
            context = torch.tensor(context).long().to(device)
            pos_pairs, neg_pairs = (torch.tensor(list(zip_longest(*item, fillvalue=0))).long().to(device).transpose(0, 1)
                                    for item in (pos_pairs, neg_pairs))  # (batch_size, longest)
            prop = torch.tensor(prop).float().to(device)

            optimizer.zero_grad()
            loss = tale_model(context, pos_pairs, neg_pairs, prop=prop)
            loss.backward()
            optimizer.step()
            trained_batches += 1
            loss_val = loss.detach().cpu().numpy().tolist()
            avg_loss += loss_val

            if trained_batches % 1000 == 0:
                lr = init_lr * (1.0 - trained_batches / batch_count)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                print('Avg loss: %.5f' % (avg_loss / 1000))
                avg_loss = 0.
    return tale_model.u_embeddings.weight.detach().cpu().numpy()
