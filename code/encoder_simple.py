import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Encoder(nn.Module):
    def __init__(
            self,
            numerical_input_dim,
            embedding_dim,
            cat_vocab_sizes,
            subseq_length):
        # only 1 categorical feature for now
        super(Encoder, self).__init__()
        self.numerical_input_dim = numerical_input_dim
        self.embedding_dim = embedding_dim
        self.cat_vocab_sizes = cat_vocab_sizes
        self.subseq_length = subseq_length

        self.num_event_encoder = nn.BatchNorm1d(numerical_input_dim)

        # why this out dim??
        self.cat_encoder = nn.Embedding(
            cat_vocab_sizes[0], int(embedding_dim/2))

        self.sequence_encoder = nn.Sequential(
            nn.GRU(numerical_input_dim+int(embedding_dim/2),
                   embedding_dim, batch_first=False)
        )

    def forward(self, n, c):
        # receives BATCH_SIZE*NUM_OF_SEQUENCES*SUBSEQUENCE_LENGTH*input_dim

        n = n.view(-1, self.numerical_input_dim)
        n = self.num_event_encoder(n)
        # n = n.view(-1, SUBSEQUENCE_LENGTH, int(self.embedding_dim/2))
        n = n.view(-1, self.subseq_length, self.numerical_input_dim)

        c = c.view(-1, 1)
        c = self.cat_encoder(c)
        c = c.view(-1, self.subseq_length, int(self.embedding_dim/2))

        x = torch.cat((n, c), 2)

        # receives BATCH_SIZE*SUBSEQUENCE_LENGTH*embedding_dim
        # so that its (seq_len, batch, input_size)
        x = torch.transpose(x, 0, 1)

        x = self.sequence_encoder(x)[0][-1]

        # normalization embedding to have unit norm
        x = x / \
            torch.linalg.norm(x, dim=1).unsqueeze(
                1).repeat(1, self.embedding_dim)

        return x
