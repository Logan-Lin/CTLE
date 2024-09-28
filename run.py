import os
from collections import Counter
from argparse import ArgumentParser

import pandas as pd

from dataset import Dataset
from downstream.loc_pre import *
from downstream.visit_time_pre import *
from downstream.traj_classify import *
from embed.ctle import CTLE, train_ctle, CTLEEmbedding, PositionalEncoding, TemporalEncoding, MaskedLM, MaskedHour
from embed.hier import HierEmbedding, Hier, train_hier
from embed.static import DownstreamEmbed, StaticEmbed
from embed.tale import TaleData, Tale, train_tale
from embed.w2v import SkipGramData, SkipGram, train_skipgram
from embed.teaser import TeaserData, Teaser, train_teaser
from embed.poi2vec import P2VData, POI2Vec

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--device', help='name of the acceleration device to use', type=str, default='cuda:0')
    parser.add_argument('--init_param', action='store_true')
    parser.add_argument('--embed_name', help='name of the embedding model to use', type=str, default='ctle')
    parser.add_argument('--embed_size', type=int, default=128)
    parser.add_argument('--embed_epoch', type=int, default=5)
    parser.add_argument('--task_name', help='name of the downstream task', type=str, default='loc_pre')
    parser.add_argument('--task_epoch', type=int, default=1)
    parser.add_argument('--task_batch_size', type=int, default=64)
    parser.add_argument('--pre_len', type=int, default=3)
    args = parser.parse_args()
    
    device = args.device
    embed_size = args.embed_size
    embed_name = args.embed_name
    embed_epoch = args.embed_epoch
    task_name = args.task_name
    task_epoch = args.task_epoch
    pre_len = args.pre_len
    init_param = args.init_param
    task_batch_size = args.task_batch_size
    
    
    hidden_size = embed_size * 4
    
    raw_df = pd.read_hdf(os.path.join('datasets', 'pek.h5'), key='data')
    coor_df = pd.read_hdf(os.path.join('datasets', 'pek.h5'), key='coor')
    
    split_days = [list(range(10, 13)), [13], [14]]
    dataset = Dataset(raw_df, coor_df, split_days)
    max_seq_len = Counter(dataset.df['user_index'].to_list()).most_common(1)[0][1]
    id2coor_df = dataset.df[['loc_index', 'lat', 'lng']].drop_duplicates('loc_index').set_index('loc_index').sort_index()
    
    
    embed_mat = np.random.uniform(low=-0.5/embed_size, high=0.5/embed_size, size=(dataset.num_loc, embed_size))
    embed_layer = StaticEmbed(embed_mat)
    if embed_name == 'downstream':
        embed_layer = DownstreamEmbed(dataset.num_loc, embed_size)
    
    if embed_name in ['skipgram', 'cbow', 'tale', 'teaser', 'poi2vec']:
        w2v_window_size = 1
        skipgram_neg = 5
    
        embed_train_users, embed_train_sentences, embed_train_weekdays, \
        embed_train_timestamp, _length = zip(*dataset.gen_sequence(min_len=w2v_window_size*2+1, select_days=0))
    
        if embed_name in ['skipgram', 'cbow']:
            sg_dataset = SkipGramData(embed_train_sentences)
            sg_model = SkipGram(dataset.num_loc, embed_size, cbow=(embed_name == 'cbow'))
            embed_mat = train_skipgram(sg_model, sg_dataset, window_size=w2v_window_size, num_neg=skipgram_neg,
                                       batch_size=64, num_epoch=embed_epoch, init_lr=1e-3, device=device)
        if embed_name == 'tale':
            tale_slice = 60
            tale_span = 30
            tale_indi_context = True
    
            tale_dataset = TaleData(embed_train_sentences, embed_train_timestamp, tale_slice, tale_span, indi_context=tale_indi_context)
            tale_model = Tale(dataset.num_loc, len(tale_dataset.id2index), embed_size)
            embed_mat = train_tale(tale_model, tale_dataset, w2v_window_size, batch_size=64, num_epoch=embed_epoch,
                                   init_lr=1e-3, device=device)
        if embed_name == 'teaser':
            teaser_num_ne = 0  # (number of unvisited locations)
            teaser_num_nn = 0  # (number of non-neighbor locations)
            teaser_indi_context = False
            teaser_beta = 0.0
            teaser_week_embed_size = 0
    
            coor_mat = dataset.df[['loc_index', 'lat', 'lng']].drop_duplicates('loc_index').to_numpy()
            teaser_dataset = TeaserData(embed_train_users, embed_train_sentences, embed_train_weekdays, coor_mat,
                                        num_ne=teaser_num_ne, num_nn=teaser_num_nn, indi_context=teaser_indi_context)
            teaser_model = Teaser(num_vocab=dataset.num_loc, num_user=len(dataset.user2index),
                                  embed_dimension=embed_size, week_embed_dimension=teaser_week_embed_size,
                                  beta=teaser_beta)
            embed_mat = train_teaser(teaser_model, teaser_dataset, window_size=w2v_window_size, num_neg=skipgram_neg,
                                     batch_size=64, num_epoch=embed_epoch, init_lr=1e-3, device=device)
        if embed_name == 'poi2vec':
            poi2vec_theta = 0.1
            poi2vec_indi_context = False
    
            poi2vec_data = P2VData(embed_train_sentences, id2coor_df, theta=poi2vec_theta, indi_context=poi2vec_indi_context)
            poi2vec_model = POI2Vec(dataset.num_loc, poi2vec_data.total_offset, embed_size)
            embed_mat = train_tale(poi2vec_model, poi2vec_data, w2v_window_size, batch_size=64, num_epoch=embed_epoch,
                                   init_lr=1e-3, device=device)
    
        embed_layer = StaticEmbed(embed_mat)
    
    if embed_name == 'hier':
        hier_num_layers = 3
        hier_week_embed_size = 4
        hier_hour_embed_size = 4
        hier_duration_embed_size = 4
        hier_share = False
    
        hier_embedding = HierEmbedding(embed_size, dataset.num_loc,
                                       hier_week_embed_size, hier_hour_embed_size, hier_duration_embed_size)
        hier_model = Hier(hier_embedding, hidden_size=hidden_size, num_layers=hier_num_layers, share=hier_share)
        embed_mat = train_hier(dataset, hier_model, num_epoch=embed_epoch, batch_size=64, device=device)
        embed_layer = StaticEmbed(embed_mat)
    
    if embed_name == 'ctle':
        encoding_type = 'positional'
        ctle_num_layers = 4
        ctle_num_heads = 8
        ctle_mask_prop = 0.2
        ctle_detach = False
        ctle_objective = "mlm"
        ctle_static = False
    
        encoding_layer = PositionalEncoding(embed_size, max_seq_len)
        if encoding_type == 'temporal':
            encoding_layer = TemporalEncoding(embed_size)
    
        obj_models = [MaskedLM(embed_size, dataset.num_loc)]
        if ctle_objective == "mh":
            obj_models.append(MaskedHour(embed_size))
        obj_models = nn.ModuleList(obj_models)
    
        ctle_embedding = CTLEEmbedding(encoding_layer, embed_size, dataset.num_loc)
        ctle_model = CTLE(ctle_embedding, hidden_size, num_layers=ctle_num_layers, num_heads=ctle_num_heads,
                          init_param=init_param, detach=ctle_detach)
        embed_layer = train_ctle(dataset, ctle_model, obj_models, mask_prop=ctle_mask_prop,
                                 num_epoch=embed_epoch, batch_size=64, device=device)
        if ctle_static:
            embed_mat = embed_layer.static_embed()
            embed_layer = StaticEmbed(embed_mat)
    
    
    if task_name == 'loc_pre':
        pre_model_name = 'mc'
        pre_model_seq2seq = True
        if pre_model_name == 'mc':
            pre_model = MCLocPredictor(dataset.num_loc)
            mc_next_loc_prediction(dataset, pre_model, pre_len)
        else:
            st_aux_embed_size = 16
            st_num_slots = 10
    
            if pre_model_name == 'erpp':
                pre_model = ErppLocPredictor(embed_layer, input_size=embed_size, lstm_hidden_size=hidden_size,
                                             fc_hidden_size=hidden_size, output_size=dataset.num_loc, num_layers=2, seq2seq=pre_model_seq2seq)
            elif pre_model_name == 'stlstm':
                pre_model = StlstmLocPredictor(embed_layer, num_slots=st_num_slots, aux_embed_size=st_aux_embed_size, time_thres=10800, dist_thres=0.1,
                                               input_size=embed_size, lstm_hidden_size=hidden_size,
                                               fc_hidden_size=hidden_size, output_size=dataset.num_loc, num_layers=2, seq2seq=pre_model_seq2seq)
            elif pre_model_name == 'rnn':
                pre_model = RnnLocPredictor(embed_layer, input_size=embed_size, rnn_hidden_size=hidden_size, fc_hidden_size=hidden_size,
                                            output_size=dataset.num_loc, num_layers=1, seq2seq=pre_model_seq2seq)
            else:
                pre_model = Seq2SeqLocPredictor(embed_layer, input_size=embed_size, hidden_size=hidden_size,
                                                output_size=dataset.num_loc, num_layers=2)
            loc_prediction(dataset, pre_model, pre_len=pre_len, num_epoch=task_epoch,
                           batch_size=64, device=device)
    
    if task_name == 'time_pre':
        pre_model_name = 'erpp'
        use_event_loss = True
    
        if pre_model_name == 'erpp':
            pre_model = ERPPTimePredictor(embed_layer, input_size=embed_size, lstm_hidden_size=hidden_size,
                                          hidden_size=hidden_size, output_size=dataset.num_loc, num_layers=2)
            erpp_visit_time_prediction(dataset, pre_model, pre_len=pre_len, num_epoch=task_epoch,
                                       batch_size=task_batch_size, device=device, use_event_loss=use_event_loss)
        if pre_model_name == 'rmtpp':
            pre_model = RMTPPTimePredictor(embed_layer, input_size=embed_size, lstm_hidden_size=hidden_size,
                                           hidden_size=hidden_size, output_size=dataset.num_loc, num_layers=2)
            erpp_visit_time_prediction(dataset, pre_model, pre_len=pre_len, num_epoch=task_epoch,
                                       batch_size=task_batch_size, device=device, use_event_loss=use_event_loss)
        pre_model = LSTMTimePredictor(embed_layer, input_size=embed_size, lstm_hidden_size=256,
                                      fc_hidden_size=256, output_size=dataset.num_loc, num_layers=2)
        lstm_visit_time_prediction(dataset, pre_model, num_epoch=task_epoch, batch_size=task_batch_size, device=device)
    
        num_time_slots = 48
        time_output_type = 'softmax'
        output_size = 1 if time_output_type == 'scalar' else num_time_slots
    
        pre_model = ScatterVisitTimePredictor(embed_layer, num_time_slots=num_time_slots,
                                              input_size=embed_size, lstm_hidden_size=512,
                                              fc_hidden_size=256, output_size=output_size, num_layers=2)
        scatter_visit_time_prediction(dataset, pre_model, time_output_type=time_output_type,
                                      num_epoch=task_epoch, batch_size=task_batch_size, device=device)
    
    if task_name == 'classify':
        traj_pooling_type = 'lstm'
    
        pre_model = FCTrajectoryClassifier(pooling_type=traj_pooling_type, input_size=embed_size, hidden_size=hidden_size, output_size=2)
        fc_trajectory_classify(dataset, embed_layer, pre_model, num_epoch=task_epoch, batch_size=task_batch_size, device=device)
    