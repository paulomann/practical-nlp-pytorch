from .metrics import ConfusionMatrix
from ..data import Average
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from .. import settings
import os
import time
import math

__all__ = [
    "Trainer",
    "Evaluator",
    "fit",
    "SequenceCriterion",
    "fit_lm",
    "LMTrainer",
    "LMEvaluator",
]


class Trainer:
    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    def run(self, model, labels, train, loss, batch_size):
        model.train()
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)

        cm = ConfusionMatrix(labels)

        for batch in train_loader:
            loss_value, y_pred, y_actual = self.update(model, loss, batch)
            _, best = y_pred.max(1)
            yt = y_actual.cpu().int().numpy()
            yp = best.cpu().int().numpy()
            cm.add_batch(yt, yp)

        print(cm.get_all_metrics())
        return cm

    def update(self, model, loss, batch):
        self.optimizer.zero_grad()
        x, lengths, y = batch
        lengths, perm_idx = lengths.sort(0, descending=True)
        x_sorted = x[perm_idx]
        y_sorted = y[perm_idx]
        y_sorted = y_sorted.to("cuda:0")
        inputs = (x_sorted.to("cuda:0"), lengths)
        y_pred = model(inputs)
        loss_value = loss(y_pred, y_sorted)
        loss_value.backward()
        self.optimizer.step()
        return loss_value.item(), y_pred, y_sorted


class Evaluator:
    def __init__(self):
        pass

    def run(self, model, labels, dataset, batch_size=1):
        model.eval()
        valid_loader = DataLoader(dataset, batch_size=batch_size)
        cm = ConfusionMatrix(labels)
        for batch in valid_loader:
            y_pred, y_actual = self.inference(model, batch)
            _, best = y_pred.max(1)
            yt = y_actual.cpu().int().numpy()
            yp = best.cpu().int().numpy()
            cm.add_batch(yt, yp)
        return cm

    def inference(self, model, batch):
        with torch.no_grad():
            x, lengths, y = batch
            lengths, perm_idx = lengths.sort(0, descending=True)
            x_sorted = x[perm_idx]
            y_sorted = y[perm_idx]
            y_sorted = y_sorted.to("cuda:0")
            inputs = (x_sorted.to("cuda:0"), lengths)
            y_pred = model(inputs)
            return y_pred, y_sorted


def fit(model, labels, optimizer, loss, epochs, batch_size, train, valid, test):

    trainer = Trainer(optimizer)
    evaluator = Evaluator()
    best_acc = 0.0

    for epoch in range(epochs):
        print("EPOCH {}".format(epoch + 1))
        print("=================================")
        print("Training Results")
        cm = trainer.run(model, labels, train, loss, batch_size)
        print("Validation Results")
        cm = evaluator.run(model, labels, valid)
        print(cm.get_all_metrics())
        if cm.get_acc() > best_acc:
            print("New best model {:.2f}".format(cm.get_acc()))
            best_acc = cm.get_acc()
            torch.save(
                model.state_dict(),
                os.path.join(settings.CHECKPOINT_PATH, "checkpoint.pt"),
            )
    if test:
        model.load_state_dict(
            torch.load(os.path.join(settings.CHECKPOINT_PATH, "checkpoint.pt"))
        )
        cm = evaluator.run(model, labels, test)
        print("Final result")
        print(cm.get_all_metrics())
    return cm.get_acc()


class SequenceCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.crit = nn.CrossEntropyLoss(ignore_index=0, size_average=True)

    def forward(self, inputs, targets):
        """Evaluate some loss over a sequence.

        :param inputs: torch.FloatTensor, [B, .., C] The scores from the model. Batch First
        :param targets: torch.LongTensor, The labels.

        :returns: torch.FloatTensor, The loss.
        """
        total_sz = targets.nelement() #nelement = numel
        loss = self.crit(inputs.view(total_sz, -1), targets.view(total_sz))
        return loss


class LMTrainer:
    def __init__(self, optimizer: torch.optim.Optimizer, nctx):
        self.optimizer = optimizer
        self.nctx = nctx

    def run(self, model, train_data, loss_function, batch_size=20, clip=0.25):
        avg_loss = Average("average_train_loss")
        metrics = {}
        self.optimizer.zero_grad()
        start = time.time()
        model.train()
        if hasattr(model, "init_hidden"):
            (h, c) = model.init_hidden(batch_size)
        num_steps = train_data.shape[1] // self.nctx
        for i in range(num_steps):
            x = train_data[:, i * self.nctx : (i + 1) * self.nctx]
            y = train_data[:, i * self.nctx + 1 : (i + 1) * self.nctx + 1]
            labels = y.to("cuda:0")
            inputs = x.to("cuda:0")
            hidden = (h.detach(), c.detach())
            logits, (h, c) = model(inputs, hidden)
            loss = loss_function(logits, labels)
            loss.backward()

            avg_loss.update(loss.item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            self.optimizer.step()
            self.optimizer.zero_grad()
            if (i + 1) % 100 == 0:
                print(avg_loss)

        # How much time elapsed in minutes
        elapsed = (time.time() - start) / 60
        train_token_loss = avg_loss.avg
        train_token_ppl = math.exp(train_token_loss)
        metrics["train_elapsed_min"] = elapsed
        metrics["average_train_loss"] = train_token_loss
        metrics["train_ppl"] = train_token_ppl
        return metrics


class LMEvaluator:
    def __init__(self, nctx):
        self.nctx = nctx

    def run(self, model, valid_data, loss_function, batch_size=20):
        avg_valid_loss = Average("average_valid_loss")
        start = time.time()
        model.eval()
        if hasattr(model, "init_hidden"):
            hidden = model.init_hidden(batch_size)
        metrics = {}
        num_steps = valid_data.shape[1] // self.nctx
        for i in range(num_steps):

            with torch.no_grad():
                x = valid_data[:, i * self.nctx : (i + 1) * self.nctx]
                y = valid_data[:, i * self.nctx + 1 : (i + 1) * self.nctx + 1]
                labels = y.to("cuda:0")
                inputs = x.to("cuda:0")
                logits, hidden = model(inputs, hidden)
                loss = loss_function(logits, labels)
                avg_valid_loss.update(loss.item())

        valid_token_loss = avg_valid_loss.avg
        valid_token_ppl = math.exp(valid_token_loss)

        elapsed = (time.time() - start) / 60
        metrics["valid_elapsed_min"] = elapsed

        metrics["average_valid_loss"] = valid_token_loss
        metrics["average_valid_word_ppl"] = valid_token_ppl
        return metrics


def fit_lm(model, optimizer, epochs, batch_size, nctx, train_data, valid_data):

    loss = SequenceCriterion()
    trainer = LMTrainer(optimizer, nctx)
    evaluator = LMEvaluator(nctx)
    best_acc = 0.0

    metrics = evaluator.run(model, valid_data, loss, batch_size)

    for epoch in range(epochs):

        print("EPOCH {}".format(epoch + 1))
        print("=================================")
        print("Training Results")
        metrics = trainer.run(model, train_data, loss, batch_size)
        print(metrics)
        print("Validation Results")
        metrics = evaluator.run(model, valid_data, loss, batch_size)
        print(metrics)
