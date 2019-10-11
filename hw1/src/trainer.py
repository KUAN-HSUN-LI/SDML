import os
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import json
from metrics import F1
from torch.optim.lr_scheduler import StepLR


class Trainer:
    def __init__(self, device, model, batch_size, lr, gradient_accumulation_steps=1, grad_clip=0.0):
        self.device = device
        self.model = model
        self.batch_size = batch_size
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, eps=1e-8)
        self.scheduler = StepLR(self.opt, step_size=2, gamma=0.5)
        self.criteria = torch.nn.BCEWithLogitsLoss()
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.grad_clip = grad_clip
        self.history = {'train': [], 'valid': []}

    def run_epoch(self, epoch, data, training):
        self.model.train(training)
        if training:
            description = 'Train'
            dataset = data
            shuffle = True
        else:
            description = 'Valid'
            dataset = data
            shuffle = False
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                shuffle=shuffle,
                                collate_fn=dataset.collate_fn,
                                num_workers=1)

        trange = tqdm(enumerate(dataloader), total=len(dataloader), desc=description)
        loss = 0
        f1_score = F1()
        for step, (tokens, segments, masks, doc_embs, labels) in trange:
            o_labels, batch_loss = self._run_iter(tokens, segments, masks, doc_embs, labels)
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

        self.scheduler.step()

    def _run_iter(self, tokens, segments, masks, doc_embs, labels):
        tokens = tokens.to(self.device)
        segments = segments.to(self.device)
        masks = masks.to(self.device)
        doc_embs = doc_embs.to(self.device)
        labels = labels.to(self.device)
        outputs = self.model(tokens, doc_embs, token_type_ids=segments, attention_mask=masks)
        l_loss = self.criteria(outputs, labels)
        return outputs, l_loss

    def save(self, epoch, save_dir):
        if not os.path.exists('../model/%s/' % save_dir):
            os.makedirs('../model/%s/' % save_dir)
        torch.save(self.model.state_dict(), '../model/%s/model.pkl.%d' % (save_dir, epoch))
        with open('../model/%s/history.json' % save_dir, 'w') as f:
            json.dump(self.history, f, indent=4)
