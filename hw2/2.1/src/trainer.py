from dataset import SentenceDataset
from utils import load_pkl, accuracy
from model import get_model
import os
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import json
import torch
import torch.nn as nn
torch.manual_seed(42)


class Trainer():
    def __init__(self, model, device, trainData, validData, opt, criterion, scheduler, batch_size):
        self.model = model.to(device)
        self.device = device
        self.trainData = trainData
        self.validData = validData
        self.opt = opt
        self.criterion = criterion
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.history = {'train': [], 'valid': []}

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
                                num_workers=0)

        trange = tqdm(enumerate(dataloader), total=len(dataloader), desc=description)
        loss = 0
        acc = accuracy()
        for i, (x, y, data_len, random_num) in trange:
            output, sent_loss = self.run_iter(x, y, data_len, random_num, training)
            batch_loss = sent_loss
            if training:
                self.opt.zero_grad()
                batch_loss.backward()
                self.opt.step()
            loss += batch_loss.item()
            acc.update(output, y[:, 1:], random_num)
            trange.set_postfix(loss=loss / (i + 1), acc=acc.print_word_score())
        if training:
            self.history['train'].append({'acc': acc.get_word_score(), 'loss': loss / len(trange)})
        else:
            self.history['valid'].append({'acc': acc.get_word_score(), 'loss': loss / len(trange)})

    def run_iter(self, x, y, data_len, random_nums, training):
        x = x.to(self.device)
        y = y.to(self.device)
        if training:
            output = self.model(x, y, data_len)
        else:
            with torch.no_grad():
                output = self.model(x, y, data_len, 0)
        sent = y[:, 1:]
        sent = sent.reshape(sent.shape[0] * sent.shape[1])
        output_sent = output.view(output.shape[0] * output.shape[1], output.shape[2])
        sent_loss = self.criterion(output_sent, sent)

        return output.argmax(2), sent_loss

    def save(self, epoch):
        if not os.path.exists('../model1'):
            os.makedirs('../model1')
        torch.save(self.model.state_dict(), '../model1/model.pkl.'+str(epoch))
        with open('../model1/history.json', 'w') as f:
            json.dump(self.history, f, indent=4)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def do_train(train_data_path, valid_data_path, word_dict_path, max_len, ncontrols, emb_dim, hid_dim, max_epoch, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    word_dict = load_pkl(word_dict_path)
    PAD_IDX = word_dict.word2idx("<PAD>")
    print("load data...")
    trainData = SentenceDataset(load_pkl(train_data_path), word_dict, PAD_IDX, max_len, ncontrols)
    validData = SentenceDataset(load_pkl(valid_data_path), word_dict, PAD_IDX, max_len, ncontrols)
    print("load model...")
    model = get_model(word_dict.get_len(), word_dict.get_len(), emb_dim, hid_dim, device)
    print(model)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    trainer = Trainer(model, device, trainData, validData, opt, criterion, scheduler, batch_size)

    for epoch in range(max_epoch):
        print('Epoch: {}'.format(epoch))
        trainer.run_epoch(epoch, True)
        trainer.run_epoch(epoch, False)
        trainer.scheduler.step()
        trainer.save(epoch)
