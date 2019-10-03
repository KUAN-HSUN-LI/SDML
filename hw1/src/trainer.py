import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from metrics import F1


class Trainer:
    def __init__(self, trainData, validData, device, model, opt, criteria, history):
        self.trainData = trainData
        self.validData = validData
        self.device = device
        self.model = model
        self.opt = opt
        self.criteria = criteria
        self.history = history

    def run_epoch(self, epoch, training):

        if epoch >= 5:
            self.model.embedding.weight.requires_grad = False

        self.model.train(training)
        if training:
            description = 'Train'
            dataset = self.trainData
            shuffle = True
        else:
            description = 'Valid'
            dataset = self.validData
            shuffle = False
        dataloader = DataLoader(dataset=dataset,
                                batch_size=16,
                                shuffle=shuffle,
                                collate_fn=dataset.collate_fn,
                                num_workers=1)

        trange = tqdm(enumerate(dataloader), total=len(dataloader), desc=description)
        loss = 0
        f1_score = F1()
        for i, (x, y, sent_len) in trange:
            o_labels, batch_loss = self._run_iter(x, y)
            if training:
                self.opt.zero_grad()
                batch_loss.backward()
                self.opt.step()

            loss += batch_loss.item()
            f1_score.update(o_labels.cpu(), y)

            trange.set_postfix(
                loss=loss / (i + 1), f1=f1_score.print_score())
        if training:
            self.history['train'].append({'f1': f1_score.get_score(), 'loss': loss / len(trange)})
        else:
            self.history['valid'].append({'f1': f1_score.get_score(), 'loss': loss / len(trange)})

    def _run_iter(self, x, y):
        abstract = x.to(self.device)
        labels = y.to(self.device)
        o_labels = self.model(abstract)
        l_loss = self.criteria(o_labels, labels)
        return o_labels, l_loss

    def save(self, epoch, dir):
        if not os.path.exists('../model/%s/' % dir):
            os.makedirs('../model/%s/' % dir)
        torch.save(self.model.state_dict(), '../model/%s/model.pkl.%d' % (dir, epoch))
        with open('../model/%s/history.json' % dir, 'w') as f:
            json.dump(self.history, f, indent=4)
