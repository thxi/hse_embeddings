import torch

import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
        self,
        numerical_input_dim,
        cat_vocab_sizes,
        embedding_dim,
    ):
        # only 1 categorical feature for now
        super(Encoder, self).__init__()
        self.numerical_input_dim = numerical_input_dim
        self.embedding_dim = embedding_dim
        self.cat_vocab_sizes = cat_vocab_sizes
        # TODO: experiment with out dim
        self.cat_embedding_dim = cat_vocab_sizes[0] // 2

        self.num_event_encoder = nn.BatchNorm1d(numerical_input_dim)

        self.cat_encoder = nn.Embedding(cat_vocab_sizes[0],
                                        self.cat_embedding_dim)

        self.sequence_encoder = nn.GRU(
            numerical_input_dim + self.cat_embedding_dim,
            embedding_dim,
            batch_first=False)

    def forward(self, n, c):
        # receives BATCH_SIZE*NUM_OF_SEQUENCES*SUBSEQUENCE_LENGTH*input_dim
        subseq_length = n.shape[-2]

        n = n.view(-1, self.numerical_input_dim)
        n = self.num_event_encoder(n)
        n = n.view(-1, subseq_length, self.numerical_input_dim)

        c = c.view(-1, 1)
        c = self.cat_encoder(c)
        c = c.view(-1, subseq_length, self.cat_embedding_dim)

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
