import torch
import pandas as pd
import pickle
import os
from sklearn.utils import shuffle
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
    print('[Info] Process csv...')
    # for train and valid csv
    trainset = pd.read_csv('../data/task2_trainset.csv', dtype=str)
    trainset = remove_info(trainset)
    trainset, validset = train_test_split(trainset, test_size=0.1, random_state=42)
    testset = pd.read_csv('../data/task2_public_testset.csv', dtype=str)
    testset = remove_info(testset)

    if not os.path.exists('../dataset/'):
        os.makedirs('../dataset/')
    trainset.to_csv('../dataset/trainset.csv', index=False)
    validset.to_csv('../dataset/validset.csv', index=False)
    testset.to_csv('../dataset/testset.csv', index=False)

    print('[Info] Collect words and make word dictionary...')
    preprocessor = Preprocessor()
    words = set()
    words |= preprocessor.collect_words('../dataset/trainset.csv')
    PAD_TOKEN = 0
    UNK_TOKEN = 1
    word_dict = {'<pad>': PAD_TOKEN, '<unk>': UNK_TOKEN}
    for word in words:
        word_dict[word] = len(word_dict)

    print('[Info] Load embedding...')
    embedder = Embedding('../data/glove.6B.300d.txt', words)

    print('[INFO] Make dataset...')
    train = preprocessor.get_dataset('../dataset/trainset.csv', embedder, pad_idx=PAD_TOKEN, n_workers=4)
    valid = preprocessor.get_dataset('../dataset/validset.csv', embedder, pad_idx=PAD_TOKEN, n_workers=4)
    test = preprocessor.get_dataset('../dataset/testset.csv', embedder, pad_idx=PAD_TOKEN, n_workers=4)

    print('[INFO] Save pickles...')
    with open('../dataset/embedding.pkl', 'wb') as f:
        pickle.dump(embedder, f)
    with open('../dataset/dictionary.pkl', 'wb') as f:
        pickle.dump(word_dict, f)
    with open('../dataset/trainData.pkl', 'wb') as f:
        pickle.dump(train, f)
    with open('../dataset/validData.pkl', 'wb') as f:
        pickle.dump(valid, f)
    with open('../dataset/testData.pkl', 'wb') as f:
        pickle.dump(test, f)


if __name__ == '__main__':
    main()
