from embed.w2v import *


class Teaser(nn.Module):
    def __init__(self, num_vocab, num_user, embed_dimension, week_embed_dimension, beta=2.0):
        super().__init__()
        self.__dict__.update(locals())

        self.u_embeddings = nn.Embedding(num_vocab+1, embed_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(num_vocab, embed_dimension + week_embed_dimension, sparse=True)
        self.user_embeddings = nn.Embedding(num_user, embed_dimension + week_embed_dimension, sparse=True)
        self.week_embeddings = nn.Embedding(2, week_embed_dimension, sparse=True)

        initrange = 0.5 / self.embed_dimension
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.week_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_u, pos_v, neg_v, user, weekday, neg_ne, neg_nn):
        """
        @param pos_u: positive input tokens, shape (batch_size).
        @param pos_v: positive output tokens, shape (batch_size, window_size*2).
        @param neg_v: negative output tokens, shape (batch_size, num_neg).
        @param user: user indices corresponding to input tokens, shape (batch_size)
        @param weekday: weekday indices corresponding to input tokens, shape (batch_size)
        @param neg_ne: negative unvisited locations, shape (batch_size, num_ne_neg)
        @param neg_nn: negative non-neighborhood locations, shape (batch_size, num_nn_neg)
        """
        embed_u = self.u_embeddings(pos_u)  # (batch_size, embed_size)
        embed_week = self.week_embeddings(weekday)  # (batch_size, embed_size)
        embed_cat = torch.cat([embed_u, embed_week], dim=-1)

        embed_v = self.v_embeddings(pos_v)  # (batch_size, N, 2*embed_size)
        score = torch.mul(embed_cat.unsqueeze(1), embed_v).squeeze()  # (batch_size, N, embed_size)
        score = torch.sum(score, dim=-1)  # (batch_size, N)
        score = F.logsigmoid(score)

        neg_embed_v = self.v_embeddings(neg_v)  # (batch_size, num_neg, embed_size)
        neg_score = torch.bmm(neg_embed_v, embed_cat.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-1 * neg_score)

        embed_user = self.user_embeddings(user)  # (batch_size, embed_size + week_embed_size)
        neg_embed_ne = self.v_embeddings(neg_ne)  # (batch_size, N, embed_size + week_embed_size)
        neg_embed_nn = self.v_embeddings(neg_nn)

        neg_ne_score = torch.bmm(embed_cat.unsqueeze(1) - neg_embed_ne, embed_user.unsqueeze(2)).squeeze()  # (batch_size, N)
        neg_ne_score = F.logsigmoid(neg_ne_score)
        neg_nn_score = torch.bmm(embed_cat.unsqueeze(1) - neg_embed_nn, embed_user.unsqueeze(2)).squeeze()
        neg_nn_score = F.logsigmoid(neg_nn_score)

        return -1 * (torch.sum(score) + torch.sum(neg_score) + self.beta * (torch.sum(neg_ne_score) + torch.sum(neg_nn_score)))

    def static_embed(self):
        return self.u_embeddings.weight[:self.num_vocab].detach().cpu().numpy()


class TeaserData(SkipGramData):
    def __init__(self, users, sentences, weeks, coordinates, num_ne, num_nn, indi_context, distance_threshold=0.2, sample=1e-3):
        """
        @param sentences: all users' full trajectories, shape (num_users, seq_len)
        @param weeks: weekday indices corresponding to the sentences.
        @param coordinates: coordinates of all locations, shape (num_locations, 3), each row is (loc_index, lat, lng)
        """
        super().__init__(sentences, sample)
        self.num_ne = num_ne
        self.num_nn = num_nn
        self.indi_context = indi_context

        self.users = users
        self.weeks = weeks
        all_locations = set(coordinates[:, 0].astype(int).tolist())

        # A dict mapping one location index to its non-neighbor locations.
        self.non_neighbors = {}
        for coor_row in coordinates:
            loc_index = int(coor_row[0])
            distance = coor_row[1:].reshape(1, 2) - coordinates[:, 1:]  # (num_loc, 2)
            distance = np.sqrt(np.power(distance[:, 0], 2) + np.power(distance[:, 1], 2))  # (num_loc)
            non_neighbor_indices = coordinates[:, 0][np.argwhere(distance > distance_threshold)].reshape(-1).astype(int)
            if non_neighbor_indices.shape[0] == 0:
                non_neighbor_indices = np.array([len(all_locations)], dtype=int)
            self.non_neighbors[loc_index] = non_neighbor_indices

        # A dict mapping one user index to its all unvisited locations.
        self.unvisited = {}
        for user, visited in zip(users, sentences):
            user = int(user)
            user_unvisited = all_locations - set(visited)
            self.unvisited[user] = user_unvisited & self.unvisited.get(user, all_locations)

    def gen_pos_pairs(self, window_size):
        pos_pairs = []
        for user, sentence, week in zip(self.users, self.sentences, self.weeks):
            for i in range(0, len(sentence) - (2 * window_size + 1)):
                target = sentence[i+window_size]
                target_week = 0 if week[i+window_size] in range(5) else 1
                context = sentence[i:i+window_size] + sentence[i+window_size+1:i+2*window_size+1]
                sample_ne = self.sample_unvisited(user, num_neg=self.num_ne)
                sample_nn = self.sample_non_neighbor(target, num_neg=self.num_nn)
                if self.indi_context:
                    pos_pairs += [[user, target, target_week, [c], sample_ne, sample_nn] for c in context]
                else:
                    pos_pairs.append([user, target, target_week, context, sample_ne, sample_nn])
        return pos_pairs

    def sample_unvisited(self, user, num_neg):
        return np.random.choice(np.array(list(self.unvisited[user])), size=(num_neg)).tolist()

    def sample_non_neighbor(self, target, num_neg):
        return np.random.choice(self.non_neighbors[target], size=(num_neg)).tolist()


def train_teaser(teaser_model, dataset, window_size, num_neg, batch_size, num_epoch, init_lr, device):
    teaser_model = teaser_model.to(device)
    optimizer = torch.optim.SGD(teaser_model.parameters(), lr=init_lr)

    pos_pairs = dataset.gen_pos_pairs(window_size)
    trained_batches = 0
    batch_count = math.ceil(num_epoch * len(pos_pairs) / batch_size)

    avg_loss = 0.
    for epoch in range(num_epoch):
        for pair_batch in next_batch(shuffle(pos_pairs), batch_size):
            neg_v = dataset.get_neg_v_sampling(len(pair_batch), num_neg)
            neg_v = torch.tensor(neg_v).long().to(device)

            user, pos_u, week, pos_v, neg_ne, neg_nn = zip(*pair_batch)
            user, pos_u, week, pos_v, neg_ne, neg_nn = (torch.tensor(item).long().to(device)
                                                        for item in (user, pos_u, week, pos_v, neg_ne, neg_nn))

            optimizer.zero_grad()
            loss = teaser_model(pos_u, pos_v, neg_v, user, week, neg_ne, neg_nn)
            loss.backward()
            optimizer.step()
            trained_batches += 1
            loss_val = loss.detach().cpu().numpy().tolist()
            avg_loss += loss_val

            if trained_batches % 10000 == 0:
                lr = init_lr * (1.0 - trained_batches / batch_count)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                print('Avg loss: %.5f' % (avg_loss / 10000), flush=True)
                avg_loss = 0.

    return teaser_model.static_embed()
