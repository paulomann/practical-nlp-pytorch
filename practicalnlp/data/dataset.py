from collections import Counter
from typing import *
import codecs
import torch
from torch.utils.data import TensorDataset
import re
import io
import codecs
import numpy as np
import torch.nn as nn

__all__ = ["Reader", "EmbeddingsReader"]


def whitespace_tokenizer(words: str) -> List[str]:
    return words.split()


def sst2_tokenizer(words: str) -> List[str]:
    REPLACE = {
        "'s": " 's ",
        "'ve": " 've ",
        "n't": " n't ",
        "'re": " 're ",
        "'d": " 'd ",
        "'ll": " 'll ",
        ",": " , ",
        "!": " ! ",
    }
    words = words.lower()
    words = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", words)
    for k, v in REPLACE.items():
        words = words.replace(k, v)
    return [w.strip() for w in words.split()]


class Reader:
    def __init__(
        self, files, lowercase=True, min_freq=0, tokenizer=sst2_tokenizer, vectorizer=None
    ):
        self.lowercase = lowercase
        self.tokenizer = tokenizer
        build_vocab = vectorizer is None
        self.vectorizer = vectorizer if vectorizer else self._vectorizer
        x = Counter()
        y = Counter()
        for file_name in files:
            if file_name is None:
                continue
            with codecs.open(file_name, encoding="utf-8", mode="r") as f:
                for line in f:
                    words = line.split()
                    y.update(words[0])

                    if build_vocab:
                        words = self.tokenizer(" ".join(words[1:]))
                        words = (
                            words if not self.lowercase else [w.lower() for w in words]
                        )
                        x.update(words)
        self.labels = list(y.keys())

        if build_vocab:
            x = dict(filter(lambda cnt: cnt[1] >= min_freq, x.items()))
            alpha = list(x.keys())
            alpha.sort()
            self.vocab = {w: i + 1 for i, w in enumerate(alpha)}
            self.vocab["[PAD]"] = 0

        self.labels.sort()

    def _vectorizer(self, words: List[str]) -> List[int]:
        return [self.vocab.get(w, 0) for w in words]

    def load(self, filename: str) -> TensorDataset:
        label2index = {l: i for i, l in enumerate(self.labels)}
        xs = []
        lengths = []
        ys = []
        with codecs.open(filename, encoding="utf-8", mode="r") as f:
            for line in f:
                words = line.split()
                ys.append(label2index[words[0]])
                words = self.tokenizer(" ".join(words[1:]))
                words = words if not self.lowercase else [w.lower() for w in words]
                vec = self.vectorizer(words)
                lengths.append(len(vec))
                xs.append(torch.tensor(vec, dtype=torch.long))
        x_tensor = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
        lengths_tensor = torch.tensor(lengths, dtype=torch.long)
        y_tensor = torch.tensor(ys, dtype=torch.long)
        return TensorDataset(x_tensor, lengths_tensor, y_tensor)


def init_embeddings(vocab_size, embed_dim, unif):
    return np.random.uniform(-unif, unif, (vocab_size, embed_dim))


class EmbeddingsReader:
    @staticmethod
    def from_text(filename, vocab, unif=0.25):

        with io.open(filename, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.rstrip("\n ")
                values = line.split(" ")

                if i == 0:
                    # fastText style
                    if len(values) == 2:
                        weight = init_embeddings(len(vocab), values[1], unif)
                        continue
                    # glove style
                    else:
                        weight = init_embeddings(len(vocab), len(values[1:]), unif)
                word = values[0]
                if word in vocab:
                    vec = np.asarray(values[1:], dtype=np.float32)
                    weight[vocab[word]] = vec
        if "[PAD]" in vocab:
            weight[vocab["[PAD]"]] = 0.0

        embeddings = nn.Embedding(weight.shape[0], weight.shape[1])
        embeddings.weight = nn.Parameter(torch.from_numpy(weight).float())
        return embeddings, weight.shape[1]

    @staticmethod
    def from_binary(filename, vocab, unif=0.25):
        def read_word(f):

            s = bytearray()
            ch = f.read(1)

            while ch != b" ":
                s.extend(ch)
                ch = f.read(1)
            s = s.decode("utf-8")
            # Only strip out normal space and \n not other spaces which are words.
            return s.strip(" \n")

        vocab_size = len(vocab)
        with io.open(filename, "rb") as f:
            header = f.readline()
            file_vocab_size, embed_dim = map(int, header.split())
            weight = init_embeddings(len(vocab), embed_dim, unif)
            if "[PAD]" in vocab:
                weight[vocab["[PAD]"]] = 0.0
            width = 4 * embed_dim
            for i in range(file_vocab_size):
                word = read_word(f)
                raw = f.read(width)
                if word in vocab:
                    vec = np.fromstring(raw, dtype=np.float32)
                    weight[vocab[word]] = vec
        embeddings = nn.Embedding(weight.shape[0], weight.shape[1])
        embeddings.weight = nn.Parameter(torch.from_numpy(weight).float())
        return embeddings, embed_dim

