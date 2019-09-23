import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    def __init__(self, vocabulary_size):
        super(SimpleNet, self).__init__()
        self.hidden_dim = 512
        self.sent_rnn = nn.GRU(vocabulary_size,
                               self.hidden_dim,
                               bidirectional=True,
                               batch_first=True)
        self.l1 = nn.Linear(self.hidden_dim, 4)

    def forward(self, x):
        b, s, w, e = x.shape
        x = x.view(b, s*w, e)
        x, __ = self.sent_rnn(x)
        x = x.view(b, s, w, -1)
        x = torch.max(x, dim=2)[0]
        x = x[:, :, :self.hidden_dim] + x[:, :, self.hidden_dim:]
        x = torch.max(x, dim=1)[0]
        x = torch.sigmoid(self.l1(F.relu(x)))
        return x
