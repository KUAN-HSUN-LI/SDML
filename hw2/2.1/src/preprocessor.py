import pickle
from word_dict import Word_dict
from sklearn.model_selection import train_test_split


def preprocess(path, max_len=25, training=True):
    dataset = []
    with open(path, 'r', encoding='UTF-8') as f:
        for line in f:
            line = line.strip()
            if training:
                dataset.append(['<SOS>'] + [word for word in line[:max_len]] + ['<EOS>'])
            else:
                dataset.append(line.split(' '))
    return dataset


def data2index(sent, word_dict):
    return [word_dict.word2idx(word) for word in sent]


def process_training_data(dataset, word_dict):
    datas = []
    for idx in range(len(dataset)-1):
        processed = {}
        processed['x'] = data2index(dataset[idx], word_dict)
        processed['y'] = data2index(dataset[idx+1], word_dict)
        datas.append(processed)
    return datas


def process_testing_data(dataset, word_dict):
    datas = []
    for data in dataset:
        datas.append(data2index(data, word_dict))
    return datas


def make_word_dict(dataset, max_len):
    word_dict = Word_dict(dataset, max_len)
    PAD_IDX = word_dict.word2idx("<PAD>")

    with open('../data/dictionary.pkl', 'wb') as f:
        pickle.dump(word_dict, f)

    return word_dict


def do_data(train_data_path, test_data_path, valid_split_rate, max_len):
    print("preprocessing data...")
    dataset = preprocess(train_data_path, max_len)
    testDataset = preprocess(test_data_path, training=False)
    print("making dictionary...")
    word_dict = make_word_dict(dataset, max_len)
    print("split and save...")
    data = process_training_data(dataset, word_dict)
    trainData, validData = train_test_split(data, test_size=0.2, random_state=42)
    testData = process_testing_data(testDataset, word_dict)

    with open("../data/train_data.pkl", "wb") as f:
        pickle.dump(trainData, f)

    with open("../data/valid_data.pkl", "wb") as f:
        pickle.dump(validData, f)

    with open("../data/test_data.pkl", "wb") as f:
        pickle.dump(testData, f)
