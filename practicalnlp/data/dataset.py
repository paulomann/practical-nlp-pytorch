from collections import Counter
from typing import *
import codecs
import torch
from torch.utils.data import TensorDataset
import re

__all__ = ["Reader"]

def whitespace_tokenizer(words: str) -> List[str]:
    return words.split() 

def sst2_tokenizer(words: str) -> List[str]:
    REPLACE = { "'s": " 's ",
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

    def __init__(self, files, lowercase=True, min_freq=0,
                 tokenizer=sst2_tokenizer, vectorizer=None):
        self.lowercase = lowercase
        self.tokenizer = tokenizer
        build_vocab = vectorizer is None
        self.vectorizer = vectorizer if vectorizer else self._vectorizer
        x = Counter()
        y = Counter()
        for file_name in files:
            if file_name is None:
                continue
            with codecs.open(file_name, encoding='utf-8', mode='r') as f:
                for line in f:
                    words = line.split()
                    y.update(words[0])

                    if build_vocab:
                        words = self.tokenizer(' '.join(words[1:]))
                        words = words if not self.lowercase else [w.lower() for w in words]
                        x.update(words)
        self.labels = list(y.keys())

        if build_vocab:
            x = dict(filter(lambda cnt: cnt[1] >= min_freq, x.items()))
            alpha = list(x.keys())
            alpha.sort()
            self.vocab = {w: i+1 for i, w in enumerate(alpha)}
            self.vocab['[PAD]'] = 0

        self.labels.sort()

    def _vectorizer(self, words: List[str]) -> List[int]:
        return [self.vocab.get(w, 0) for w in words]

    def load(self, filename: str) -> TensorDataset:
        label2index = {l: i for i, l in enumerate(self.labels)}
        xs = []
        lengths = []
        ys = []
        with codecs.open(filename, encoding='utf-8', mode='r') as f:
            for line in f:
                words = line.split()
                ys.append(label2index[words[0]])
                words = self.tokenizer(' '.join(words[1:]))
                words = words if not self.lowercase else [w.lower() for w in words]
                vec = self.vectorizer(words)
                lengths.append(len(vec))
                xs.append(torch.tensor(vec, dtype=torch.long))
        x_tensor = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
        lengths_tensor = torch.tensor(lengths, dtype=torch.long)
        y_tensor = torch.tensor(ys, dtype=torch.long)
        return TensorDataset(x_tensor, lengths_tensor, y_tensor)
