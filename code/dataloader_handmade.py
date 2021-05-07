from pathlib import Path
import torch
import pickle
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

NUM_OF_SUBSEQUENCES = 5  # number of sequences per person
SUBSEQUENCE_LENGTH = 90


class AgeGroupHandmadeDataset(torch.utils.data.Dataset):
    def __init__(self,
                 subseq_length=SUBSEQUENCE_LENGTH,
                 num_of_subseq=NUM_OF_SUBSEQUENCES,
                 root=Path('data/')):
        self.client_df = pd.read_csv(root / 'client_df.csv', index_col=0)

        self.X = self.client_df.drop('bins', axis=1).reset_index(drop=True)
        self.y = self.client_df.reset_index()['bins']

        self.X = torch.from_numpy(self.X.to_numpy()).type(torch.float32)
        self.y = torch.from_numpy(self.y.to_numpy()).type(torch.int)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
