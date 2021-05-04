from pathlib import Path

import torch
import pickle
import pandas as pd
import numpy as np
from tqdm.auto import tqdm


def normalize(col):
    return (col - col.mean())/col.std()


def process_transactions(df):
    shifted = (
        df
        .sort_values(['client_id', 'trans_date'])
        .groupby('client_id')['trans_date'].shift(1)
    ).reset_index()
    df['prev_date'] = shifted['trans_date']
    # drop first transaction for each client since it does not have the prev date
    df = df[~shifted['trans_date'].isna()].copy()
    df['prev_diff'] = df['trans_date'] - df['prev_date']
    df.drop('prev_date', axis=1, inplace=True)
    df['weekday'] = df['trans_date'] % 7
    for c in ['amount_rur', 'weekday', 'trans_date', 'prev_diff']:
        df[c] = normalize(df[c])
    return df


NUM_OF_SUBSEQUENCES = 5  # number of sequences per person
SUBSEQUENCE_LENGTH = 90


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        subseq_length=SUBSEQUENCE_LENGTH,
        num_of_subseq=NUM_OF_SUBSEQUENCES,
        root=Path('../data/')
    ):
        super().__init__()
        self.root = root
        self.subseq_length = subseq_length
        self.num_of_subseq = num_of_subseq

        raw_df = pd.read_csv(root/'transactions_train.csv')

        # NOTE: remapping client_ids to indices
        self.idx2client = raw_df['client_id'].unique()

        raw_df['client_id'] = raw_df['client_id'].map(
            {v: k for k, v in enumerate(self.idx2client)})

        self.df = process_transactions(raw_df)
        self.df = self.df.sort_values(['client_id', 'trans_date'])

        self.numerical = torch.from_numpy(self.df.drop(
            'client_id', axis=1).to_numpy()).type(torch.float32)
        self.categorical = torch.from_numpy(
            self.df[['small_group']].to_numpy()).type(torch.int)

    def __len__(self):
        return len(self.client_to_indices)

    def get_sequence(self, idx):
        s, e = self.client_to_indices[idx]
        n = self.numerical[s:e]
        c = self.categorical[s:e]
        return n, c

    def build_client_to_indices(self):
        self.client_to_indices = {}  # tuple of (index start, index after end)
        # so that numpy_array[1:end] yields the sequence
        current = self.df.iloc[0, 0]
        start_idx = 0
        print('building client_to_indices', len(self.df))
        for i, (_, r) in tqdm(enumerate(self.df.iterrows())):
            if current != r['client_id']:
                self.client_to_indices[current] = (start_idx, i)
                start_idx = i
                current = r['client_id']

        self.client_to_indices[current] = (start_idx, i+1)
        print('finished buildiing client_to_indices')

    def load_client_to_indices(self):
        self.client_to_indices = pickle.load(
            open(self.root/"client_to_indices.p", "rb"))

    def save_client_to_indices(self):
        pickle.dump(self.client_to_indices,
                    open(self.root/"client_to_indices.p", "wb"))


class AgeGroupMLDataset(BaseDataset):
    def __init__(
        self,
        subseq_length=SUBSEQUENCE_LENGTH,
        num_of_subseq=NUM_OF_SUBSEQUENCES,
        root=Path('../data/')
    ):
        super().__init__(subseq_length, num_of_subseq, root)

        self.targets = list(range(len(self.idx2client)))

    def __getitem__(self, idx: int):
        # outputs ((n, c), idx)
        # dim(n,c) are  self.num_of_subseq x self.subseq_length x feature_dim

        n, c = self.get_sequence(idx)

        sn = torch.zeros((self.num_of_subseq, self.subseq_length, n.size(1))).type(
            torch.float32)
        sc = torch.zeros(
            (self.num_of_subseq, self.subseq_length, c.size(1))).type(torch.int)

        seq_len = n.size(0)
        for i in range(self.num_of_subseq):
            start_index = np.random.randint(0, seq_len-self.subseq_length+1)
            end_index = start_index + self.subseq_length
            nsubseq = n[start_index:end_index]
            csubseq = c[start_index:end_index]

            sn[i] = nsubseq
            sc[i] = csubseq

        return (sn, sc), idx


class AgeGroupClfDataset(BaseDataset):
    def __init__(
        self,
        subseq_length=SUBSEQUENCE_LENGTH,
        num_of_subseq=NUM_OF_SUBSEQUENCES,
        root=Path('../data/')
    ):
        super().__init__(subseq_length, num_of_subseq, root)

        target_df = pd.read_csv(root/'train_target.csv')
        target_df['client_id'] = target_df['client_id'].map(
            {v: k for k, v in enumerate(self.idx2client)})
        self.target_df = target_df

        self.targets = self.target_df.sort_values('client_id')[
            'bins'].to_numpy()

    def __getitem__(self, idx: int):
        # returns ((numerical, categorical), target_bin)
        n, c = self.get_sequence(idx)

        seq_len = len(n)
        start_index = np.random.randint(0, seq_len-SUBSEQUENCE_LENGTH+1)
        end_index = start_index + SUBSEQUENCE_LENGTH
        n = n[start_index:end_index]
        c = c[start_index:end_index]

        return (n, c), self.targets[idx]
