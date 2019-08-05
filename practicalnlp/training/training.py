from .metrics import ConfusionMatrix
import torch
from torch.utils.data.dataloader import DataLoader
from .. import settings
import os

__all__ = ["Trainer, Evaluator", "fit"]

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
        y_sorted = y_sorted.to('cuda:0')
        inputs = (x_sorted.to('cuda:0'), lengths)
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
            y_sorted = y_sorted.to('cuda:0')
            inputs = (x_sorted.to('cuda:0'), lengths)
            y_pred = model(inputs)
            return y_pred, y_sorted

def fit(model, labels, optimizer, loss, epochs, batch_size, train, valid, test):

    trainer = Trainer(optimizer)
    evaluator = Evaluator()
    best_acc = 0.0
    
    for epoch in range(epochs):
        print('EPOCH {}'.format(epoch + 1))
        print('=================================')
        print('Training Results')
        cm = trainer.run(model, labels, train, loss, batch_size)
        print('Validation Results')
        cm = evaluator.run(model, labels, valid)
        print(cm.get_all_metrics())
        if cm.get_acc() > best_acc:
            print('New best model {:.2f}'.format(cm.get_acc()))
            best_acc = cm.get_acc()
            torch.save(model.state_dict(),
                       os.path.join(settings.CHECKPOINT_PATH, "checkpoint.pt"))
    if test:
        model.load_state_dict(torch.load(
            os.path.join(settings.CHECKPOINT_PATH, "checkpoint.pt")))
        cm = evaluator.run(model, labels, test)
        print('Final result')
        print(cm.get_all_metrics())
    return cm.get_acc()