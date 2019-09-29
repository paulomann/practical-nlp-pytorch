import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import copy

__all__ = [
    "LSTMClassifier",
    "ConvClassifier",
    "LSTMLanguageModel",
    "TransformerEncoderStack",
    "TransformerEncoder",
    "MultiHeadedAttention",
    "PositionalEncoding",
    "TransformerLM"
]


# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------ HANDY FUNCTIONS -------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------


def clone_module(module_, N):
    return nn.ModuleList([copy.deepcopy(module_) for _ in range(N)])


def subsequent_mask(size):
    """
    Creates a lower triangular mask to mask future
    :param size: Temporal length
    :return: A tensor of type `uint8` that is 1s along diagonals and below, zero  o.w
    """
    attn_shape = (1, 1, size, size)
    sub_mask = np.tril(np.ones(attn_shape)).astype("uint8")
    return torch.from_numpy(sub_mask)


def pytorch_linear(in_sz, out_sz, unif=0, initializer=None):
    l = nn.Linear(in_sz, out_sz)
    if unif > 0:
        l.weight.data.uniform_(-unif, unif)
    elif initializer == "ortho":
        nn.init.orthogonal(l.weight)
    elif initializer == "he" or initializer == "kaiming":
        nn.init.kaiming_uniform(l.weight)
    else:
        nn.init.xavier_uniform_(l.weight)

    l.bias.data.zero_()
    return l


# ------------------------------------------------------------------
# ------------------------------------------------------------------
# -------------------------- CLASSIFIERS ---------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------


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
        # decoded size = torch.Size([20, 35, 512])
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


# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------- TRANSFORMER ----------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------


def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """Scaled dot product attention, as defined in https://arxiv.org/abs/1706.03762

    We apply the query to the keys to recieve our weights via softmax, which are then applied
    for each value, but in a series of efficient matrix operations.  In the case of self-attention,
    the key, query and values are all low order projections of the same input.

    :param query: a query for alignment. Can come from self in case of self-attn or decoder in case of E/D
    :param key: a set of keys from encoder or self
    :param value: a set of values from encoder or self
    :param mask: masking (for destination) to prevent seeing what we shouldnt
    :param dropout: apply dropout operator post-attention (this is not a float)
    :return: A tensor that is (BxHxTxT)

    """
    # (., H, T, T) = (., H, T, D) x (., H, D, T)
    d_k = query.size(-1)
    # (BATCH_SIZE, N_HEADS, T, d_k)
    # This scores calculation happens to be the multiplication between the "fixed"
    # part of both query and key dimensions, i.e., (BATCH_SIZE, N_HEADS). This tuple
    # could be seen as a batch, while the (d_k, T) is multiplied between query and key
    # In other words, the matrix multiplication is always done with using the last
    # two dimensions. All the ones before are considered as batch.
    # query x key.transpose(-2, -1) == (BATCH, N_HEADS, T, d_k) x (BATCH, N_HEADS, d_k, T)
    # and the result of this matmul is (BATCH, N_HEADS, T, T) because only matrices
    # (T, d_k) and (d_k, T) are multiplied batch-wise.
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = F.softmax(scores, dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    return torch.matmul(weights, value), weights


class MultiHeadedAttention(nn.Module):
    """
    Multi-headed attention from https://arxiv.org/abs/1706.03762 via http://nlp.seas.harvard.edu/2018/04/03/attention.html

    Multi-headed attention provides multiple looks of low-order projections K, Q and V using an attention function
    (specifically `scaled_dot_product_attention` in the paper.  This allows multiple relationships to be illuminated
    via attention on different positional and representational information from each head.

    The number of heads `h` times the low-order projection dim `d_k` is equal to `d_model` (which is asserted upfront).
    This means that each weight matrix can be simply represented as a linear transformation from `d_model` to `d_model`,
    and partitioned into heads after the fact.

    Finally, an output projection is applied which brings the output space back to `d_model`, in preparation for the
    sub-sequent `FFN` sub-layer.

    There are 3 uses of multi-head attention in the Transformer.
    For encoder-decoder layers, the queries come from the previous decoder layer, and the memory keys come from
    the encoder.  For encoder layers, the K, Q and V all come from the output of the previous layer of the encoder.
    And for self-attention in the decoder, K, Q and V all come from the decoder, but here it is masked to prevent using
    future values
    """

    def __init__(self, h, d_model, dropout=0.1):
        """Constructor for multi-headed attention

        :param h: The number of heads
        :param d_model: The model hidden size
        :param dropout (``float``): The amount of dropout to use
        :param attn_fn: A function to apply attention, defaults to SDP
        """
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.w_Q = nn.Linear(d_model, d_model)
        self.w_K = nn.Linear(d_model, d_model)
        self.w_V = nn.Linear(d_model, d_model)
        self.w_O = nn.Linear(d_model, d_model)
        self.attn_fn = scaled_dot_product_attention
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """Low-order projections of query, key and value into multiple heads, then attention application and dropout

        :param query: a query for alignment. Can come from self in case of self-attn or decoder in case of E/D
        :param key: a set of keys from encoder or self
        :param value: a set of values from encoder or self
        :param mask: masking (for destination) to prevent seeing what we shouldnt
        :return: Multi-head attention output, result of attention application to sequence (B, T, d_model)
        """
        batchsz = query.size(0)

        # (B, H, T, D)
        query = self.w_Q(query).view(batchsz, -1, self.h, self.d_k).transpose(1, 2)
        key = self.w_K(key).view(batchsz, -1, self.h, self.d_k).transpose(1, 2)
        value = self.w_V(value).view(batchsz, -1, self.h, self.d_k).transpose(1, 2)

        x, self.attn = self.attn_fn(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(batchsz, -1, self.h * self.d_k)
        return self.w_O(x)


class TransformerEncoder(nn.Module):
    def __init__(self, num_heads, d_model, pdrop, d_ff=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.self_attn = MultiHeadedAttention(num_heads, d_model, pdrop)
        self.ffn = nn.Sequential(
            pytorch_linear(self.d_model, self.d_ff),
            nn.ReLU(),
            pytorch_linear(self.d_ff, self.d_model),
        )
        self.ln1 = nn.LayerNorm(self.d_model, eps=1e-12)
        self.ln2 = nn.LayerNorm(self.d_model, eps=1e-12)
        self.dropout = nn.Dropout(pdrop)

    def forward(self, x, mask=None):
        """
        :param x:
        :param mask:
        :return:
        """
        # Builtin Attention mask
        x = self.ln1(x)
        h = self.self_attn(x, x, x, mask)
        x = x + self.dropout(h)

        x = self.ln2(x)
        x = x + self.dropout(self.ffn(x))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, num_heads, d_model, pdrop, d_ff=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.self_attn = MultiHeadedAttention(num_heads, self.d_model, pdrop)
        self.src_attn = MultiHeadedAttention(num_heads, self.d_model, pdrop)
        self.ffn = nn.Sequential(
            pytorch_linear(self.d_model, self.d_ff),
            nn.ReLU(),
            pytorch_linear(self.d_ff, self.d_model),
        )

        self.ln1 = nn.LayerNorm(self.d_model, eps=1e-12)
        self.ln2 = nn.LayerNorm(self.d_model, eps=1e-12)
        self.ln3 = nn.LayerNorm(self.d_model, eps=1e-12)
        self.dropout = nn.Dropout(pdrop)

    def forward(self, x, memory, src_mask, tgt_mask):

        x = self.ln1(x)
        x = x + self.dropout(self.self_attn(x, x, x, tgt_mask))

        x = self.ln2(x)
        x = x + self.dropout(self.src_attn(x, memory, memory, src_mask))

        x = self.ln3(x)
        x = x + self.dropout(self.ffn(x))
        return x


class TransformerEncoderStack(nn.Module):
    def __init__(self, num_heads, d_model, pdrop, layers=1, d_ff=None):
        super().__init__()
        single_layer = TransformerEncoder(num_heads, d_model, pdrop, d_ff)
        self.layers = clone_module(single_layer, layers)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerDecoderStack(nn.Module):
    def __init__(self, num_heads, d_model, pdrop, layers=1, d_ff=None):
        super().__init__()
        single_layer = TransformerDecoder(num_heads, d_model, pdrop, d_ff)
        self.layers = clone_module(single_layer, layers)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # register_buffer impede the weights of pe to be trainable by the optimizer
        # but it saves and is restored in the state_dict
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerEmbedding(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class Generator(nn.Module):
    "Standard generation step."

    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class TransformerLM(nn.Module):
    """
    This transformer LM is similar to GPT
    """

    def __init__(self, vocab_size, d_model=512, h=8, pdrop=0.1, layers=12, d_ff=None):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.position = PositionalEncoding(d_model, pdrop)
        self.encoder_stack = TransformerEncoderStack(
            num_heads=h, d_model=d_model, pdrop=pdrop, layers=layers, d_ff=d_ff
        )
        self.generator = Generator(d_model=d_model, vocab=vocab_size)
        self.embedding = TransformerEmbedding(d_model=d_model, vocab=vocab_size)
        # We tie weights
        self.generator.proj.weight = self.embedding.lut.weight
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.encoder_stack(x, mask)
        x = self.generator(x)
        return x
