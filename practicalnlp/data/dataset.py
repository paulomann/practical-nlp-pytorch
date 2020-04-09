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
from tqdm import tqdm
import itertools
from torch.utils.data import Dataset
import torch

__all__ = ["Reader", "EmbeddingsReader", "WordDatasetReader", "Average"]


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
        self,
        files,
        lowercase=True,
        min_freq=0,
        tokenizer=sst2_tokenizer,
        vectorizer=None,
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
                            words
                            if not self.lowercase
                            else [w.lower() for w in words]
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
                words = (
                    words if not self.lowercase else [w.lower() for w in words]
                )
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
                        weight = init_embeddings(
                            len(vocab), len(values[1:]), unif
                        )
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


class WordDatasetReader(object):
    """Provide a base-class to do operations to read words to tensors
    """

    def __init__(self, nctx, vectorizer=None):
        self.nctx = nctx
        self.num_words = {}
        self.vectorizer = vectorizer if vectorizer else self._vectorizer

    def build_vocab(self, files, min_freq=0):
        x = Counter()

        for file in files:
            if file is None:
                continue
            self.num_words[file] = 0
            with codecs.open(file, encoding="utf-8", mode="r") as f:
                sentences = []
                for line in f:
                    split_sentence = line.split() + ["<EOS>"]
                    self.num_words[file] += len(split_sentence)
                    sentences += split_sentence
                x.update(Counter(sentences))
        x = dict(filter(lambda cnt: cnt[1] >= min_freq, x.items()))
        alpha = list(x.keys())
        alpha.sort()
        self.vocab = {w: i + 1 for i, w in enumerate(alpha)}
        self.vocab["[PAD]"] = 0

    def _vectorizer(self, words: List[str]) -> List[int]:
        return [self.vocab.get(w, 0) for w in words]

    def load_features(self, filename):

        with codecs.open(filename, encoding="utf-8", mode="r") as f:
            sentences = []
            for line in f:
                sentences += line.strip().split() + ["<EOS>"]
            return torch.tensor(self.vectorizer(sentences), dtype=torch.long)

    def load(self, filename, batch_size):
        x_tensor = self.load_features(filename)
        rest = x_tensor.shape[0] // batch_size
        num_steps = rest // self.nctx
        # if num_examples is divisible by batchsz * nctx (equivalent to rest is divisible by nctx), we
        # have a problem. reduce rest in that case.

        if rest % self.nctx == 0:
            rest = rest - 1
        trunc = batch_size * rest

        x_tensor = x_tensor.narrow(0, 0, trunc)
        # torch.Size([20, 104431]) for nctx = 35 and batch_size = 20
        x_tensor = x_tensor.view(batch_size, -1).contiguous()
        return x_tensor


class Average(object):
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


####################################################################################################
####################################################################################################
############################### TSP DATASET FOR THE POINTER NETWORKS ###############################
####################################################################################################
####################################################################################################


def tsp_opt(points):
    """
    Dynamic programing solution for TSP - O(2^n*n^2)
    https://gist.github.com/mlalevic/6222750
    :param points: List of (x, y) points
    :return: Optimal solution
    """

    def length(x_coord, y_coord):
        return np.linalg.norm(np.asarray(x_coord) - np.asarray(y_coord))

    # Calculate all lengths
    all_distances = [[length(x, y) for y in points] for x in points]
    # Initial value - just distance from 0 to every other point + keep the track of edges
    A = {
        (frozenset([0, idx + 1]), idx + 1): (dist, [0, idx + 1])
        for idx, dist in enumerate(all_distances[0][1:])
    }
    cnt = len(points)
    for m in range(2, cnt):
        B = {}
        for S in [
            frozenset(C) | {0} for C in itertools.combinations(range(1, cnt), m)
        ]:
            for j in S - {0}:
                # This will use 0th index of tuple for ordering, the same as if key=itemgetter(0) used
                B[(S, j)] = min(
                    [
                        (
                            A[(S - {j}, k)][0] + all_distances[k][j],
                            A[(S - {j}, k)][1] + [j],
                        )
                        for k in S
                        if k != 0 and k != j
                    ]
                )
        A = B
    res = min([(A[d][0] + all_distances[0][d[1]], A[d][1]) for d in iter(A)])
    return np.asarray(res[1])


class TSPDataset(Dataset):
    """
    Random TSP dataset
    """

    def __init__(self, data_size, seq_len, solver=tsp_opt, solve=True):
        self.data_size = data_size
        self.seq_len = seq_len
        self.solve = solve
        self.solver = solver
        self.data = self._generate_data()

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        tensor = torch.from_numpy(self.data["Points_List"][idx]).float()
        solution = (
            torch.from_numpy(self.data["Solutions"][idx]).long()
            if self.solve
            else None
        )

        sample = {"Points": tensor, "Solution": solution}

        return sample

    def _generate_data(self):
        """
        :return: Set of points_list ans their One-Hot vector solutions
        """
        points_list = []
        solutions = []
        data_iter = tqdm(range(self.data_size), unit="data")
        for i, _ in enumerate(data_iter):
            data_iter.set_description(
                "Data points %i/%i" % (i + 1, self.data_size)
            )
            points_list.append(np.random.random((self.seq_len, 2)))
        solutions_iter = tqdm(points_list, unit="solve")
        if self.solve:
            for i, points in enumerate(solutions_iter):
                solutions_iter.set_description(
                    "Solved %i/%i" % (i + 1, len(points_list))
                )
                solutions.append(self.solver(points))
        else:
            solutions = None

        return {"Points_List": points_list, "Solutions": solutions}

    def _to1hotvec(self, points):
        """
        :param points: List of integers representing the points indexes
        :return: Matrix of One-Hot vectors
        """
        vec = np.zeros((len(points), self.seq_len))
        for i, v in enumerate(vec):
            v[points[i]] = 1

        return vec
