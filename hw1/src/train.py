import torch
import torch.nn as nn
import pickle
import json
import matplotlib.pyplot as plt
from network import SimpleNet
from trainer import Trainer
import argparse


def plot(dir):
    with open('../%s/history.json' % dir, 'r') as f:
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
    plt.savefig('../%s/loss.png' % dir)

    plt.figure(figsize=(7, 5))
    plt.title('F1 Score')
    plt.plot(train_f1, label='train')
    plt.plot(valid_f1, label='valid')
    plt.legend()
    plt.savefig('../%s/f1_score.png' % dir)

    print('Best F1 score ', max([[l['f1'], idx] for idx, l in enumerate(history['valid'])]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir_name', type=str, default='model')
    parser.add_argument('--cuda', type=int, default=0)
    args = parser.parse_args()

    with open('../dataset/embedding.pkl', 'rb') as f:
        embedder = pickle.load(f)
    with open('../dataset/trainData.pkl', 'rb') as f:
        trainData = pickle.load(f)
    with open('../dataset/validData.pkl', 'rb') as f:
        validData = pickle.load(f)

    device = torch.device('cuda:%s' % args.cuda if torch.cuda.is_available() else 'cpu')
    model = SimpleNet(embedder.get_dim())
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1.0e-4)
    criteria = torch.nn.BCELoss()
    model.to(device)
    max_epoch = 25
    history = {'train': [], 'valid': []}

    embedding = nn.Embedding(embedder.get_vocabulary_size(), embedder.get_dim())
    embedding.weight = torch.nn.Parameter(embedder.vectors)

    trainer = Trainer(trainData, validData, device, model, opt, criteria, history)
    for epoch in range(max_epoch):
        print('Epoch: {}'.format(epoch))
        trainer.run_epoch(epoch, embedding, True)
        trainer.run_epoch(epoch, embedding, False)
        trainer.save(epoch, args.save_dir_name)
    plot(args.save_dir_name)


if __name__ == '__main__':
    main()
