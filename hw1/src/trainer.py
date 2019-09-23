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

    def run_epoch(self, epoch, embedding, training):
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
                                batch_size=32,
                                shuffle=shuffle,
                                collate_fn=dataset.collate_fn,
                                num_workers=4)

        trange = tqdm(enumerate(dataloader), total=len(dataloader), desc=description)
        loss = 0
        f1_score = F1()
        for i, (x, y, sent_len) in trange:
            x = embedding(x)
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

    def save(self, epoch):
        if not os.path.exists('../model/'):
            os.makedirs('../model/')
        torch.save(self.model.state_dict(), '../model/model.pkl.' + str(epoch))
        with open('../model/history.json', 'w') as f:
            json.dump(self.history, f, indent=4)
