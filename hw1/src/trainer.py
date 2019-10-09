import os
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import json
from metrics import F1


class Trainer:
    def __init__(self, batch_size, trainData, validData,
                 device, model, opt, criteria, history, gradient_accumulation_steps=1, grad_clip=0.0):
        self.batch_size = batch_size
        self.trainData = trainData
        self.validData = validData
        self.device = device
        self.model = model
        self.opt = opt
        self.criteria = criteria
        self.history = history

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.grad_clip = grad_clip

    def run_epoch(self, epoch, training):
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
                                batch_size=self.batch_size,
                                shuffle=shuffle,
                                collate_fn=dataset.collate_fn,
                                num_workers=1)

        trange = tqdm(enumerate(dataloader), total=len(dataloader), desc=description)
        loss = 0
        f1_score = F1()
        for step, (tokens, segments, masks, labels) in trange:
            o_labels, batch_loss = self._run_iter(tokens, segments, masks, labels)
            if training:
                if self.gradient_accumulation_steps > 1:
                    batch_loss = batch_loss / self.gradient_accumulation_steps
                batch_loss.backward()
                # clip_grad_norm_(self.model.parameters(), self.grad_clip)
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.opt.step()
                    self.opt.zero_grad()

            loss += batch_loss.item()
            f1_score.update(o_labels.cpu(), labels)

            trange.set_postfix(
                loss=loss / (step + 1), f1=f1_score.print_score())
        if training:
            self.history['train'].append({'f1': f1_score.get_score(), 'loss': loss / len(trange)})
        else:
            self.history['valid'].append({'f1': f1_score.get_score(), 'loss': loss / len(trange)})

    def _run_iter(self, tokens, segments, masks, labels):
        tokens = tokens.to(self.device)
        segments = segments.to(self.device)
        masks = masks.to(self.device)
        labels = labels.to(self.device)
        outputs = self.model(tokens, token_type_ids=segments, attention_mask=masks)
        l_loss = self.criteria(outputs, labels)
        return outputs, l_loss

    def save(self, epoch):
        if not os.path.exists('../model/'):
            os.makedirs('../model/')
        torch.save(self.model.state_dict(), '../model/model.pkl.%d' % epoch)
        with open('../model/history.json', 'w') as f:
            json.dump(self.history, f, indent=4)
