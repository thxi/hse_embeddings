import torch
import torch.nn as nn
from pytorch_tabnet.tab_network import TabNet


class Encoder(nn.Module):
    def __init__(
        self,
        numerical_input_dim,
        cat_vocab_sizes,
        cat_embedding_dim,
        embedding_dim,
    ):
        # only 1 categorical feature for now
        super(Encoder, self).__init__()
        self.numerical_input_dim = numerical_input_dim
        self.embedding_dim = embedding_dim
        self.cat_vocab_sizes = cat_vocab_sizes
        # TODO: experiment with out dim
        self.cat_embedding_dim = cat_embedding_dim

        self.num_event_encoder = nn.BatchNorm1d(numerical_input_dim)

        self.sequence_encoder = nn.GRU(embedding_dim,
                                       embedding_dim,
                                       batch_first=False)
        self.tabnet = TabNet(input_dim=numerical_input_dim +
                             len(cat_vocab_sizes),
                             output_dim=embedding_dim,
                             n_d=8,
                             n_a=8,
                             n_steps=3,
                             gamma=1.3,
                             cat_idxs=[4],
                             cat_dims=cat_vocab_sizes,
                             cat_emb_dim=cat_embedding_dim,
                             n_independent=2,
                             n_shared=2,
                             epsilon=1e-15,
                             virtual_batch_size=128,
                             momentum=0.02,
                             mask_type="sparsemax")

    def forward(self, n, c):
        # receives BATCH_SIZE*NUM_OF_SEQUENCES*SUBSEQUENCE_LENGTH*input_dim
        subseq_length = n.shape[-2]

        # just in case you pass subsequences only (without NUM_OF_SEQUENCES dim)
        tmp = len(n.shape)
        x = torch.cat((n, c), tmp - 1)

        x = x.view(-1, self.numerical_input_dim + len(self.cat_vocab_sizes))

        # print(x.shape)
        x, _ = self.tabnet(x)
        x = x.view(-1, subseq_length, self.embedding_dim)

        # receives BATCH_SIZE*SUBSEQUENCE_LENGTH*embedding_dim
        # so that its (seq_len, batch, input_size)
        x = torch.transpose(x, 0, 1)

        x = self.sequence_encoder(x)[0][-1]

        # normalization embedding to have unit norm
        x = x / \
            torch.linalg.norm(x, dim=1).unsqueeze(
                1).repeat(1, self.embedding_dim)

        return x
