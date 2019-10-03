import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    def __init__(self, embedder):
        super(SimpleNet, self).__init__()
        self.hidden_dim = 256
        self.sent_rnn = nn.GRU(embedder.get_dim(),
                               self.hidden_dim,
                               bidirectional=True,
                               batch_first=True)
        self.l1 = nn.Linear(self.hidden_dim, 4)

        self.embedding = nn.Embedding(embedder.get_vocabulary_size(), embedder.get_dim())
        self.embedding.weight = nn.Parameter(embedder.vectors)

        self.dropout = nn.Dropout(0.5)
        self.layerNorm = nn.LayerNorm([self.hidden_dim * 2])

    def forward(self, x):
        x = self.embedding(x)
        b, s, w, e = x.shape
        x = x.view(b, s*w, e)
        x, __ = self.sent_rnn(x)
        # x = self.layerNorm(x)
        x = self.dropout(x)
        x = x.view(b, s, w, -1)
        x = torch.max(x, dim=2)[0]
        x = x[:, :, :self.hidden_dim] + x[:, :, self.hidden_dim:]
        x = torch.max(x, dim=1)[0]
        x = self.l1(F.relu(x))
        x = torch.sigmoid(x)
        return x
