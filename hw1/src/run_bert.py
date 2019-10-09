import torch
import pandas as pd
import pickle
import os
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("--do_data", action='store_true')
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument('--pretrained_model_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_len', default=256, type=int)
    parser.add_argument('--epochs', default=6, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--cuda', default=1, type=int)
    parser.add_argument('--checkpoint', default=-1, type=int)
    args = parser.parse_args()

    if args.do_data:
        preprocess(args)

    if args.do_train:
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

    print('[Info] Collect words and make word dictionary...')
    preprocessor = Preprocessor(args.pretrained_model_name)

    print('[INFO] Make dataset...')
    train_data = preprocessor.get_dataset(trainset, args.max_len, n_workers=4)
    valid_data = preprocessor.get_dataset(validset, args.max_len, n_workers=4)
    test_data = preprocessor.get_dataset(testset, args.max_len, n_workers=4)

    print('[INFO] Save pickles...')
    if not os.path.exists('../dataset/'):
        os.makedirs('../dataset/')
    with open('../dataset/trainData_%d.pkl' % args.max_len, 'wb') as f:
        pickle.dump(train_data, f)
    with open('../dataset/validData_%d.pkl' % args.max_len, 'wb') as f:
        pickle.dump(valid_data, f)
    with open('../dataset/testData_%d.pkl' % args.max_len, 'wb') as f:
        pickle.dump(test_data, f)


def train(args):
    from trainer import Trainer
    from network import BertForMultiLabelSequenceClassification
    from utils import plot

    with open('../dataset/trainData_%d.pkl' % args.max_len, 'rb') as f:
        train_data = pickle.load(f)
    with open('../dataset/validData_%d.pkl' % args.max_len, 'rb') as f:
        valid_data = pickle.load(f)

    device = torch.device('cuda:%d' % args.cuda if torch.cuda.is_available() else 'cpu')
    model = BertForMultiLabelSequenceClassification.from_pretrained(args.pretrained_model_name, num_labels=4)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, eps=1e-8)
    criteria = torch.nn.BCEWithLogitsLoss()

    history = {'train': [], 'valid': []}

    trainer = Trainer(args.batch_size, train_data, valid_data, device, model, opt, criteria, history)
    for epoch in range(args.epochs):
        print('Epoch: {}'.format(epoch))
        if epoch > 1:
            model.freeze_bert_encoder()
        trainer.run_epoch(epoch, True)
        trainer.run_epoch(epoch, False)
        trainer.save(epoch)
    plot()


def predict(args):
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from network import BertForMultiLabelSequenceClassification
    from utils import SubmitGenerator

    with open('../dataset/testData_%d.pkl' % args.max_len, 'rb') as f:
        testData = pickle.load(f)

    device = torch.device('cuda:%d' % args.cuda if torch.cuda.is_available() else 'cpu')
    model = BertForMultiLabelSequenceClassification.from_pretrained(args.pretrained_model_name, num_labels=4)
    model.load_state_dict(torch.load('../model/model.pkl.{}'.format(args.checkpoint)))
    model.train(False)
    model.to(device)

    dataloader = DataLoader(dataset=testData,
                            batch_size=args.batch_size,
                            shuffle=False,
                            collate_fn=testData.collate_fn,
                            num_workers=1)
    trange = tqdm(enumerate(dataloader), total=len(dataloader), desc='Predict')
    prediction = []
    for i, (tokens, segments, masks, labels) in trange:
        with torch.no_grad():
            o_labels = model(tokens.to(device), segments.to(device), masks.to(device))
            o_labels = o_labels > 0.5
            prediction.append(o_labels.to('cpu'))

    prediction = torch.cat(prediction).detach().numpy().astype(int)

    if not os.path.exists('../score/task 2/'):
        os.makedirs('../score/task 2/')
    SubmitGenerator(prediction, '../data/task2_sample_submission.csv', True, '../score/task 2/task2_submission.csv')


if __name__ == '__main__':
    main()
