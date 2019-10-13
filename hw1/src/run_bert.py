import torch
import pandas as pd
import pickle
import os
from argparse import ArgumentParser
from tfidf import get_tfidf
from dataset import BertDataset
import ipdb
from embedding import Embedding
from utils import collect_words


def main():
    parser = ArgumentParser()
    parser.add_argument('--dir_name', type=str, required=True)
    parser.add_argument("--do_data", action='store_true')
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument('--model', default='bert-large-uncased', type=str, help='pretrained_model_name')
    parser.add_argument('--max_len', default=256, type=int)
    parser.add_argument('--epochs', default=6, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--accum', default=1, type=int, help='gradient_accumulation_steps')
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--cuda', default=-1, type=int)
    parser.add_argument('--checkpoint', default=-1, type=int)
    parser.add_argument('--fz', default=-1, type=int, help='freeze bert epoch')
    parser.add_argument('--tfidf', action='store_true')
    args = parser.parse_args()

    data_name = '%s_%d' % (args.model.split('-', 1)[1], args.max_len)
    if args.tfidf:
        data_name += '_tfidf'

    if args.do_data:
        preprocess(args, data_name)

    if args.do_train:
        if not os.path.exists('../dataset/trainData_%s.pkl' % data_name):
            preprocess(args, data_name)
        train(args, data_name)

    if args.do_test:
        predict(args, data_name)


def preprocess(args, data_name):
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

    preprocessor = Preprocessor(args.model, None)

    print('[INFO] Get data')
    train = preprocessor.get_dataset(trainset, n_workers=12)
    valid = preprocessor.get_dataset(validset, n_workers=12)
    test = preprocessor.get_dataset(testset, n_workers=12)

    if args.tfidf:
        print('[Info] Collect words and make embedding...')
        words = set()
        words |= collect_words(trainset)
        words |= collect_words(validset)
        words |= collect_words(testset)
        embedding = Embedding('../data/glove.6B.300d.txt', words)
        preprocessor.embedding = embedding

        train_tfidf = preprocessor.get_tfidf(trainset, n_workers=12)
        valid_tfidf = preprocessor.get_tfidf(validset, n_workers=12)
        test_tfidf = preprocessor.get_tfidf(testset, n_workers=12)

        print('[INFO] Make abstract tfidf document embedding...')
        tfidf = get_tfidf(train_tfidf + valid_tfidf + test_tfidf)
        train_doc_emb = preprocessor.get_glove_tfidf_emb(trainset, tfidf[:6300])
        valid_doc_emb = preprocessor.get_glove_tfidf_emb(validset, tfidf[6300:7000])
        test_doc_emb = preprocessor.get_glove_tfidf_emb(testset, tfidf[7000:])

        print('[INFO] Make bert dataset...')
        train_data = BertDataset(train, train_doc_emb, args.max_len)
        valid_data = BertDataset(valid, valid_doc_emb, args.max_len)
        test_data = BertDataset(test, test_doc_emb, args.max_len)
    else:
        train_data = BertDataset(train, None, args.max_len)
        valid_data = BertDataset(valid, None, args.max_len)
        test_data = BertDataset(test, None, args.max_len)

    print('[INFO] Save pickles...')
    if not os.path.exists('../dataset/'):
        os.makedirs('../dataset/')
    with open('../dataset/trainData_%s.pkl' % data_name, 'wb') as f:
        pickle.dump(train_data, f)
    with open('../dataset/validData_%s.pkl' % data_name, 'wb') as f:
        pickle.dump(valid_data, f)
    with open('../dataset/testData_%s.pkl' % data_name, 'wb') as f:
        pickle.dump(test_data, f)


def train(args, data_name):
    from trainer import Trainer
    from network import BertForMultiLabelSequenceClassification
    from utils import plot

    with open('../dataset/trainData_%s.pkl' % data_name, 'rb') as f:
        train_data = pickle.load(f)
    with open('../dataset/validData_%s.pkl' % data_name, 'rb') as f:
        valid_data = pickle.load(f)

    device = torch.device('cuda:%d' % args.cuda if torch.cuda.is_available() else 'cpu')
    # model = BertForMultiLabelSequenceClassification.from_pretrained(args.model, num_labels=3, output_hidden_states=True)
    model = BertForMultiLabelSequenceClassification.from_pretrained(args.model, num_labels=3)
    model.to(device)

    trainer = Trainer(device, model, args.batch_size, args.lr, args.accum, args.grad_clip, args.fz)

    for epoch in range(args.epochs):
        print('Epoch: {}'.format(epoch))
        trainer.run_epoch(epoch, train_data, True)
        trainer.run_epoch(epoch, valid_data, False)
        trainer.save(epoch, args.dir_name)
    plot(args.dir_name)


def predict(args, data_name):
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from network import BertForMultiLabelSequenceClassification
    from utils import SubmitGenerator

    with open('../dataset/testData_%s.pkl' % data_name, 'rb') as f:
        testData = pickle.load(f)

    device = torch.device('cuda:%d' % args.cuda if torch.cuda.is_available() else 'cpu')
    model = BertForMultiLabelSequenceClassification.from_pretrained(args.model, num_labels=3)
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
    for i, (tokens, segments, masks, doc_embs, labels) in trange:
        with torch.no_grad():
            o_labels = model(tokens.to(device), doc_embs.to(device), device, segments.to(device), masks.to(device))
            o_labels = o_labels > 0.0
            prediction.append(o_labels.to('cpu'))

    prediction = torch.cat(prediction).detach().numpy().astype(int)

    if not os.path.exists('../score/task 2/'):
        os.makedirs('../score/task 2/')
    SubmitGenerator(prediction, '../data/task2_sample_submission.csv', True, '../score/task 2/task2_submission.csv')


if __name__ == '__main__':
    main()
