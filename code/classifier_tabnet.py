import torch.nn as nn
from pytorch_tabnet.tab_network import TabNet


class Classifier(nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        out_dim,
    ):
        super(Classifier, self).__init__()
        self.tabnet = TabNet(input_dim=input_dim,
                             output_dim=embedding_dim,
                             n_d=32,
                             n_a=32,
                             n_steps=5,
                             gamma=1.3,
                             n_independent=2,
                             n_shared=2,
                             epsilon=1e-15,
                             virtual_batch_size=128,
                             momentum=0.02,
                             mask_type="sparsemax")
        self.fc = nn.Linear(embedding_dim, out_dim)

    def forward(self, x):
        x, _ = self.tabnet(x)
        x = self.fc(x)
        return x
