import torch.nn as nn
from code.encoder_gru import Encoder
from code.decoder import Decoder


# todo: decoder NN
class Classifier(nn.Module):
    def __init__(
        self,
        numerical_input_dim,
        cat_vocab_sizes,
        cat_embedding_dim,
        embedding_dim,
    ):
        super(Classifier, self).__init__()
        self.encoder = Encoder(
            numerical_input_dim=numerical_input_dim,
            cat_vocab_sizes=cat_vocab_sizes,
            cat_embedding_dim=cat_embedding_dim,
            embedding_dim=embedding_dim,
        )
        self.decoder = Decoder(embedding_dim)

        self.encoder_frozen = False

    def forward(self, n, c):
        embeddings = self.encoder(n, c)
        outputs = self.decoder(embeddings)
        return outputs

    def block(self, in_dim, out_dim):
        return nn.Sequential(nn.Linear(in_dim, out_dim), nn.Sigmoid(),
                             nn.BatchNorm1d(out_dim))

    def freeze_encoder(self):
        # freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.encoder.eval()

    def unfreeze_encoder(self):
        # freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = True

        self.encoder.train()

    def train_decoder(self):
        self.decoder.train()

    def eval_decoder(self):
        self.decoder.eval()
