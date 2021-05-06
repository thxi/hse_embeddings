import torch.nn as nn


# todo: decoder NN
class Decoder(nn.Module):
    def __init__(self, input_dim):
        super(Decoder, self).__init__()
        self.input_dim = input_dim

        self.decoder = nn.Sequential(
            self.block(input_dim, 64),
            self.block(64, 16),
            self.block(16, 4),
            nn.Linear(4, 4),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

    def block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Sigmoid(),
            nn.BatchNorm1d(out_dim))
