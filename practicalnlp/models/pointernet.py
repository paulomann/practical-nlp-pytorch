import torch
import torch.nn as nn
from practicalnlp.data import TSPDataset


__all__ = ["PointerNet"]


# Extracted code from fairseq [1]
# https://github.com/pytorch/fairseq/blob/e75cff5f2c1d62f12dc911e0bf420025eb1a4e33/fairseq/models/lstm.py#L471
# def Embedding(num_embeddings, embedding_dim, padding_idx):
#     m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
#     nn.init.uniform_(m.weight, -0.1, 0.1)
#     nn.init.constant_(m.weight[padding_idx], 0)
#     return m


# def LSTM(input_size, hidden_size, **kwargs):
#     m = nn.LSTM(input_size, hidden_size, **kwargs)
#     for name, param in m.named_parameters():
#         if 'weight' in name or 'bias' in name:
#             param.data.uniform_(-0.1, 0.1)
#     return m


# def LSTMCell(input_size, hidden_size, **kwargs):
#     m = nn.LSTMCell(input_size, hidden_size, **kwargs)
#     for name, param in m.named_parameters():
#         if 'weight' in name or 'bias' in name:
#             param.data.uniform_(-0.1, 0.1)
#     return m


# def Linear(in_features, out_features, bias=True, dropout=0):
#     """Linear layer (input: N x T x C)"""
#     m = nn.Linear(in_features, out_features, bias=bias)
#     m.weight.data.uniform_(-0.1, 0.1)
#     if bias:
#         m.bias.data.uniform_(-0.1, 0.1)
#     return m

class Encoder(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, lstm_layers: int, dropout: float, bidir: bool):
        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.bidir = bidir
        self.rnn = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.lstm_layers,
            dropout=self.dropout,
            bidirectional=self.bidir
        )
        self._init_weights()
    
    def forward(self):
        pass
    
    def _init_weights(self):
        """ Uniform initialization is commonly employed by
        official PyTorch LM tutorial [1] and by Fairseq [2]
        both using -0.1 and 0.1 for the min and max params.
        [1] https://github.com/pytorch/examples/blob/ad775ace1b9db09146cdd0724ce9195f7f863fff/word_language_model/model.py#L42
        [2] https://github.com/pytorch/fairseq/blob/e75cff5f2c1d62f12dc911e0bf420025eb1a4e33/fairseq/models/lstm.py#L478 """
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.uniform_(param.data, -0.1, 0.1)
            if "bias" in name:
                nn.init.zeros_(param.data)

    def init_hidden(self, bsz):
        """ Same init_hidden code from [1]
        [1] https://github.com/pytorch/examples/blob/ad775ace1b9db09146cdd0724ce9195f7f863fff/word_language_model/model.py#L56 """
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)


class Decoder(nn.Module):
    pass

class Attention(nn.Module):
    pass

class PointerNet(nn.Module):
    pass