import os

import numpy as np
import pandas as pd

from utils import gen_index_map


class Dataset:
    def __init__(self, raw_df, coor_df, split_days):
        """
        @param raw_df: raw DataFrame containing all mobile signaling records.
            Should have at least three columns: user_id, latlng and datetime.
        @param coor_df: DataFrame containing coordinate information.
            With an index corresponding to latlng, and two columns: lat and lng.
        """
        self.latlng2index = gen_index_map(raw_df, 'latlng')
        self.user2index = gen_index_map(raw_df, 'user_id')
        raw_df['loc_index'] = raw_df['latlng'].map(self.latlng2index)
        raw_df['user_index'] = raw_df['user_id'].map(self.user2index)
        raw_df = raw_df.merge(coor_df, on='latlng', how='left')
        self.df = raw_df[['user_index', 'loc_index', 'datetime', 'lat', 'lng']]

        self.num_user = len(self.user2index)
        self.num_loc = len(self.latlng2index)
        self.split_days = split_days

    def gen_sequence(self, min_len=0, select_days=None, include_delta=False):
        """
        Generate moving sequence from original trajectories.

        @param min_len: minimal length of sentences.
        @param select_day: list of day to select, set to None to use all days.
        """
        data = pd.DataFrame(self.df, copy=True)
        data['day'] = data['datetime'].dt.day
        if select_days is not None:
            data = data[data['day'].isin(self.split_days[select_days])]
        data['weekday'] = data['datetime'].dt.weekday
        data['timestamp'] = data['datetime'].apply(lambda x: x.timestamp())

        if include_delta:
            data['time_delta'] = data['timestamp'].shift(-1) - data['timestamp']
            coor_delta = (data[['lng', 'lat']].shift(-1) - data[['lng', 'lat']]).to_numpy()
            data['dist'] = np.sqrt((coor_delta ** 2).sum(-1))

        seq_set = []
        for (user_index, day), group in data.groupby(['user_index', 'day']):

            if group.shape[0] < min_len:
                continue
            one_set = [user_index, group['loc_index'].tolist(), group['weekday'].astype(int).tolist(),
                       group['timestamp'].astype(int).tolist(), group.shape[0]]

            if include_delta:
                one_set += [[0] + group['time_delta'].iloc[:-1].tolist(),
                            [0] + group['dist'].iloc[:-1].tolist(),
                            group['lat'].tolist(),
                            group['lng'].tolist()]

            seq_set.append(one_set)
            # if user_index >= 10: break
        return seq_set

    def gen_span(self, span_len, select_day=None):
        data = pd.DataFrame(self.df, copy=True)
        data['day'] = data['datetime'].dt.day
        if select_day is not None:
            data = data[data['day'].isin(select_day)]
        data['weekday'] = data['datetime'].dt.weekday
        data['timestamp'] = data['datetime'].apply(lambda x: x.timestamp())

        seq_set = []
        for (user_index, day), group in data.groupby(['user_index', 'day']):
            for i in range(group.shape[0] - span_len + 1):
                select_group = group.iloc[i:i+span_len]
                one_set = [user_index, select_group['loc_index'].tolist(), select_group['weekday'].astype(int).tolist(),
                           select_group['timestamp'].astype(int).tolist()]
                seq_set.append(one_set)
        return seq_set


if __name__ == '__main__':
    raw_df = pd.read_hdf(os.path.join('data', 'sy.h5'), key='data')
    coor_df = pd.read_hdf(os.path.join('data', 'sy.h5'), key='coor')
    dataset = Dataset(raw_df, coor_df)
    seq_set = dataset.gen_sequence(0)
    pass
