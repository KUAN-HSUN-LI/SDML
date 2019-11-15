from torch.nn.utils.rnn import pack_padded_sequence
from torch import nn
import torch


class Embedding(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(Embedding, self).__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)

    def forward(self, input):
        embedded = self.embedding(input)

        return embedded


class Encoder(nn.Module):
    def __init__(self, emb_dim, hid_dim, embedding):
        super(Encoder, self).__init__()

        self.embedding = embedding

        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True)

    def forward(self, input, data_len):
        input = self.embedding(input)

        pack_datas = pack_padded_sequence(input, data_len, batch_first=True)

        outputs, hidden = self.rnn(input)

        return hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, embedding):
        super(Decoder, self).__init__()

        self.output_dim = output_dim

        self.embedding = embedding

        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True)

        self.out = nn.Linear(hid_dim, output_dim)

    def forward(self, input, hidden):
        input = self.embedding(input)

        output, hidden = self.rnn(input, hidden)

        prediction = self.out(output)

        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.device = device

    def forward(self, src, trg=None, data_len=None, teacher_forcing_ratio=0.5, training=True):
        batch_size = src.shape[0]
        if training:
            max_len = trg.shape[1]
        else:
            max_len = 25

        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, max_len-1, trg_vocab_size).to(self.device)

        hidden = self.encoder(src, data_len)

        input = src[:, 0:1]
        for t in range(1, max_len):
            output, hidden = self.decoder(input, hidden)
            outputs[:, t-1:t] = output

            top1 = output.argmax(2)

            if torch.rand(1) > teacher_forcing_ratio:
                input = top1
            else:
                input = trg[:, t:t+1]
        return outputs


def get_model(input_dim, output_dim, emb_dim, hid_dim, device):
    emb = Embedding(input_dim, emb_dim)
    enc = Encoder(emb_dim, hid_dim, emb)
    dec = Decoder(output_dim, emb_dim, hid_dim, emb)

    model = Seq2Seq(enc, dec, device)

    return model
