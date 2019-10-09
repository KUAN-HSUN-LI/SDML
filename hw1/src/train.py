import torch
import torch.nn as nn
import pickle
import json
import matplotlib.pyplot as plt
from network import BertForMultiLabelSequenceClassification
from trainer import Trainer
import argparse


def plot():
    with open('../model/history.json', 'r') as f:
        history = json.loads(f.read())

    train_loss = [l['loss'] for l in history['train']]
    valid_loss = [l['loss'] for l in history['valid']]
    train_f1 = [l['f1'] for l in history['train']]
    valid_f1 = [l['f1'] for l in history['valid']]

    plt.figure(figsize=(7, 5))
    plt.title('Loss')
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='valid')
    plt.legend()
    plt.savefig('../model/loss.png')

    plt.figure(figsize=(7, 5))
    plt.title('F1 Score')
    plt.plot(train_f1, label='train')
    plt.plot(valid_f1, label='valid')
    plt.legend()
    plt.savefig('../model/f1_score.png')

    print('Best F1 score ', max([[l['f1'], idx] for idx, l in enumerate(history['valid'])]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--max_len', default=256, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--epochs', default=6, type=int)
    parser.add_argument('--cuda', default=1, type=int)
    args = parser.parse_args()

    with open('../dataset/trainData_%d.pkl' % args.max_len, 'rb') as f:
        trainData = pickle.load(f)
    with open('../dataset/validData_%d.pkl' % args.max_len, 'rb') as f:
        validData = pickle.load(f)

    device = torch.device('cuda:%d' % args.cuda if torch.cuda.is_available() else 'cpu')
    model = BertForMultiLabelSequenceClassification.from_pretrained(args.pretrained_model_name, num_labels=4)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, eps=1e-8)
    criteria = torch.nn.BCEWithLogitsLoss()

    history = {'train': [], 'valid': []}

    trainer = Trainer(args.batch_size, trainData, validData, device, model, opt, criteria, history)
    for epoch in range(args.epochs):
        print('Epoch: {}'.format(epoch))
        if epoch > 1:
            model.freeze_bert_encoder()
        trainer.run_epoch(epoch, True)
        trainer.run_epoch(epoch, False)
        trainer.save(epoch)
    plot()


if __name__ == '__main__':
    main()
