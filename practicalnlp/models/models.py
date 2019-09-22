import torch.nn as nn
import torch
import torch.nn.functional as F

__all__ = ["LSTMClassifier", "ConvClassifier", "LSTMLanguageModel"]


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        embeddings,
        num_classes,
        embed_dims,
        rnn_units,
        rnn_layers=1,
        dropout=0.5,
        hidden_units=[],
    ):
        super().__init__()
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)
        self.rnn = torch.nn.LSTM(
            embed_dims,
            rnn_units,
            rnn_layers,
            dropout=dropout,
            bidirectional=False,
            batch_first=False,
        )
        nn.init.orthogonal_(self.rnn.weight_hh_l0)
        nn.init.orthogonal_(self.rnn.weight_ih_l0)
        sequence = []
        input_units = rnn_units
        output_units = rnn_units
        for h in hidden_units:
            sequence.append(nn.Linear(input_units, h))
            input_units = h
            output_units = h

        sequence.append(nn.Linear(output_units, num_classes))
        self.outputs = nn.Sequential(*sequence)

    def forward(self, inputs):
        one_hots, lengths = inputs
        embed = self.dropout(self.embeddings(one_hots))
        embed = embed.permute(1, 0, 2)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embed, lengths.tolist())
        _, hidden = self.rnn(packed)
        hidden = hidden[0].view(hidden[0].shape[1:])
        linear = self.outputs(hidden)
        return F.log_softmax(linear, dim=-1)


# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------CONVOLUTIONAL NEURAL NETWORKS-------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------


class ParallelConv(nn.Module):
    def __init__(self, input_dims, filters, dropout=0.5):
        super().__init__()
        convs = []
        self.output_dims = sum([t[1] for t in filters])
        for (filter_length, output_dims) in filters:
            pad = filter_length // 2
            conv = nn.Sequential(
                nn.Conv1d(
                    in_channels=input_dims,
                    out_channels=output_dims,
                    kernel_size=filter_length,
                    padding=pad,
                ),
                nn.ReLU(),
            )
            convs.append(conv)
        # Add the module so its managed correctly
        self.convs = nn.ModuleList(convs)
        self.conv_drop = nn.Dropout(dropout)

    def forward(self, input_bct):
        mots = []
        for conv in self.convs:
            # In Conv1d, data BxCxT, max over time
            conv_out = conv(input_bct)
            mot, _ = conv_out.max(2)
            mots.append(mot)
        mots = torch.cat(mots, 1)
        return self.conv_drop(mots)


class ConvClassifier(nn.Module):
    def __init__(
        self,
        embeddings,
        num_classes,
        embed_dims,
        filters=[(2, 100), (3, 100), (4, 100)],
        dropout=0.5,
        hidden_units=[],
    ):
        super().__init__()
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)
        self.convs = ParallelConv(embed_dims, filters, dropout)

        input_units = self.convs.output_dims
        output_units = self.convs.output_dims
        sequence = []
        for h in hidden_units:
            sequence.append(self.dropout(nn.Linear(input_units, h)))
            input_units = h
            output_units = h

        sequence.append(nn.Linear(output_units, num_classes))
        self.outputs = nn.Sequential(*sequence)

    def forward(self, inputs):
        one_hots, _ = inputs
        embed = self.dropout(self.embeddings(one_hots))
        embed = embed.transpose(1, 2).contiguous()
        hidden = self.convs(embed)
        linear = self.outputs(hidden)
        return F.log_softmax(linear, dim=-1)

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ---------------------- Language Modeling -------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout=0.5, layers=2):
        super().__init__()
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = torch.nn.LSTM(
            embed_dim,
            hidden_dim,
            layers,
            dropout=dropout,
            bidirectional=False,
            batch_first=True,
        )
        self.proj = nn.Linear(embed_dim, vocab_size)
        self.proj.bias.data.zero_()

        # Tie weights
        self.proj.weight = self.embed.weight

    def forward(self, x, hidden):
        # emb size = torch.Size([20, 35, 512])
        emb = self.embed(x)
        decoded, hidden = self.rnn(emb, hidden)
        #decoded size = torch.Size([20, 35, 512])
        return self.proj(decoded), hidden

    def init_hidden(self, batchsz):
        weight = next(self.parameters()).data
        return (
            torch.autograd.Variable(
                weight.new(self.layers, batchsz, self.hidden_dim).zero_()
            ),
            torch.autograd.Variable(
                weight.new(self.layers, batchsz, self.hidden_dim).zero_()
            ),
        )
