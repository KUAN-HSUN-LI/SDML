import torch
import pandas as pd
import pickle
import os
import argparse
from sklearn.model_selection import train_test_split

from preprocessor import Preprocessor
from embedding import Embedding
torch.manual_seed(42)


def remove_info(dataset):
    dataset.drop('Title', axis=1, inplace=True)
    dataset.drop('Categories', axis=1, inplace=True)
    dataset.drop('Created Date', axis=1, inplace=True)
    dataset.drop('Authors', axis=1, inplace=True)
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--max_len', type=int, default=256)
    args = parser.parse_args()

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
    trainData = preprocessor.get_dataset(trainset, args.max_len, n_workers=4)
    validData = preprocessor.get_dataset(validset, args.max_len, n_workers=4)
    testData = preprocessor.get_dataset(testset, args.max_len, n_workers=4)

    print('[INFO] Save pickles...')
    if not os.path.exists('../dataset/'):
        os.makedirs('../dataset/')
    with open('../dataset/trainData_%d.pkl' % args.max_len, 'wb') as f:
        pickle.dump(trainData, f)
    with open('../dataset/validData_%d.pkl' % args.max_len, 'wb') as f:
        pickle.dump(validData, f)
    with open('../dataset/testData_%d.pkl' % args.max_len, 'wb') as f:
        pickle.dump(testData, f)


if __name__ == '__main__':
    main()
