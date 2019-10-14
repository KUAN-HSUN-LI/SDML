import torch
import pandas as pd
import pickle
import os
from argparse import ArgumentParser
from utils import load_pkl
from tfidf import get_tfidf
from dataset import BertDataset
import ipdb


def main():
    parser = ArgumentParser()
    parser.add_argument('--dir_name', type=str, required=True)
    parser.add_argument("--do_data", action='store_true')
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument('--model', default='bert-base-uncased', type=str, help='pretrained_model_name')
    parser.add_argument('--max_len', default=256, type=int)
    parser.add_argument('--epochs', default=6, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--accum', default=1, type=int, help='gradient_accumulation_steps')
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--cuda', default=-1, type=int)
    parser.add_argument('--checkpoint', default=-1, type=int)
    parser.add_argument('--fz', default=-1, type=int)
    args = parser.parse_args()

    global data_name
    data_name = '%s_%d_nodeVec_tfidf' % (args.model.split('-', 1)[1], args.max_len)
    if args.do_data:
        preprocess(args)

    if args.do_train:
        if not os.path.exists('../dataset/trainData_%s.pkl' % data_name):
            preprocess(args)
        train(args)

    if args.do_test:
        predict(args)


def preprocess(args):
    from utils import remove_info
    from sklearn.model_selection import train_test_split
    from preprocessor import Preprocessor
    torch.manual_seed(42)

    print('[Info] Process csv...')
    # for train and valid csv
    trainset = pd.read_csv('../data/task2_trainset.csv', dtype=str)
    trainset = remove_info(trainset)
    trainset, validset = train_test_split(trainset, test_size=0.1, random_state=42)
    testset = pd.read_csv('../data/task2_public_testset.csv', dtype=str)
    testset = remove_info(testset)

    print('[Info] Loading node vectors...')
    train_node_vec = load_pkl('../data/node_vec.pkl')     # torch([7000, 128]
    train_node_vec, valid_node_vec = train_test_split(train_node_vec.numpy(), test_size=0.1, random_state=42)
    train_node_vec = torch.FloatTensor(train_node_vec)
    valid_node_vec = torch.FloatTensor(valid_node_vec)
    test_node_vec = load_pkl('../data/node_vec_test.pkl')
    test_node_vec = test_node_vec.type(torch.FloatTensor)

    print('[INFO] Make bert dataset...')
    preprocessor = Preprocessor(args.model)
    train_data = preprocessor.get_dataset(trainset, n_workers=12)
    valid_data = preprocessor.get_dataset(validset, n_workers=12)
    test_data = preprocessor.get_dataset(testset, n_workers=12)

    tfidf = get_tfidf([data['tokens'] for data in train_data] +
                      [data['tokens'] for data in valid_data] +
                      [data['tokens'] for data in test_data])

    train_data = BertDataset(train_data, train_node_vec, tfidf[:6300], args.max_len)
    valid_data = BertDataset(valid_data, valid_node_vec, tfidf[6300:7000], args.max_len)
    test_data = BertDataset(test_data, test_node_vec, tfidf[7000:], args.max_len)

    print('[INFO] Save pickles...')
    if not os.path.exists('../dataset/'):
        os.makedirs('../dataset/')
    with open('../dataset/trainData_%s.pkl' % data_name, 'wb') as f:
        pickle.dump(train_data, f)
    with open('../dataset/validData_%s.pkl' % data_name, 'wb') as f:
        pickle.dump(valid_data, f)
    with open('../dataset/testData_%s.pkl' % data_name, 'wb') as f:
        pickle.dump(test_data, f)


def train(args):
    from trainer import Trainer
    from network import BertForMultiLabelSequenceClassification
    from utils import plot

    with open('../dataset/trainData_%s.pkl' % data_name, 'rb') as f:
        train_data = pickle.load(f)
    with open('../dataset/validData_%s.pkl' % data_name, 'rb') as f:
        valid_data = pickle.load(f)

    device = torch.device('cuda:%d' % args.cuda if torch.cuda.is_available() else 'cpu')
    model = BertForMultiLabelSequenceClassification.from_pretrained(args.model, num_labels=4)
    model.to(device)

    trainer = Trainer(device, model, args.batch_size, args.lr, args.accum, args.grad_clip)

    for epoch in range(args.epochs):
        print('Epoch: {}'.format(epoch))
        trainer.run_epoch(epoch, train_data, True)
        trainer.run_epoch(epoch, valid_data, False)
        trainer.save(epoch, args.dir_name)
    plot(args.dir_name)


def predict(args):
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from network import BertForMultiLabelSequenceClassification
    from utils import SubmitGenerator

    with open('../dataset/testData_%s.pkl' % data_name, 'rb') as f:
        testData = pickle.load(f)

    device = torch.device('cuda:%d' % args.cuda if torch.cuda.is_available() else 'cpu')
    model = BertForMultiLabelSequenceClassification.from_pretrained(args.model, num_labels=4)
    model.load_state_dict(torch.load('../model/%s/model.pkl.%d' % (args.dir_name, args.checkpoint)))
    model.train(False)
    model.to(device)

    dataloader = DataLoader(dataset=testData,
                            batch_size=args.batch_size,
                            shuffle=False,
                            collate_fn=testData.collate_fn,
                            num_workers=1)
    trange = tqdm(enumerate(dataloader), total=len(dataloader), desc='Predict')
    prediction = []
    for i, (tokens, segments, masks, outputs, labels) in trange:
        with torch.no_grad():
            o_labels = model(tokens.to(device), outputs.to(device), segments.to(device), masks.to(device))
            o_labels = o_labels > 0.0
            prediction.append(o_labels.to('cpu'))

    prediction = torch.cat(prediction).detach().numpy().astype(int)

    if not os.path.exists('../score/task 2/'):
        os.makedirs('../score/task 2/')
    SubmitGenerator(prediction, '../data/task2_sample_submission.csv', True, '../score/task 2/task2_submission.csv')


if __name__ == '__main__':
    main()
