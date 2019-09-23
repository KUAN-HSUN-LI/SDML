import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
from network import SimpleNet
import pandas as pd
import os


def SubmitGenerator(prediction, sampleFile, public=True, filename='prediction.csv'):
    """
    Args:
        prediction (numpy array)
        sampleFile (str)
        public (boolean)
        filename (str)
    """
    sample = pd.read_csv(sampleFile)
    submit = {}
    submit['order_id'] = list(sample.order_id.values)
    redundant = len(sample) - prediction.shape[0]
    if public:
        submit['THEORETICAL'] = list(prediction[:, 0]) + [0]*redundant
        submit['ENGINEERING'] = list(prediction[:, 1]) + [0]*redundant
        submit['EMPIRICAL'] = list(prediction[:, 2]) + [0]*redundant
        submit['OTHERS'] = list(prediction[:, 3]) + [0]*redundant
    else:
        submit['THEORETICAL'] = [0]*redundant + list(prediction[:, 0])
        submit['ENGINEERING'] = [0]*redundant + list(prediction[:, 1])
        submit['EMPIRICAL'] = [0]*redundant + list(prediction[:, 2])
        submit['OTHERS'] = [0]*redundant + list(prediction[:, 3])
    df = pd.DataFrame.from_dict(submit)
    df.to_csv(filename, index=False)


def main():
    with open('../dataset/embedding.pkl', 'rb') as f:
        embedder = pickle.load(f)
    with open('../dataset/testData.pkl', 'rb') as f:
        testData = pickle.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleNet(embedder.get_dim())
    model.load_state_dict(torch.load('../model/model.pkl.{}'.format(22)))
    model.train(False)
    model.to(device)
    embedding = nn.Embedding(embedder.get_vocabulary_size(), embedder.get_dim())
    embedding.weight = torch.nn.Parameter(embedder.vectors)

    dataloader = DataLoader(dataset=testData,
                            batch_size=128,
                            shuffle=False,
                            collate_fn=testData.collate_fn,
                            num_workers=2)
    trange = tqdm(enumerate(dataloader), total=len(dataloader), desc='Predict')
    prediction = []
    for i, (x, y, _l) in trange:
        x = embedding(x)
        o_labels = model(x.to(device))
        o_labels = o_labels > 0.5
        prediction.append(o_labels.to('cpu'))

    prediction = torch.cat(prediction).detach().numpy().astype(int)

    if not os.path.exists('../score/task 2/'):
        os.makedirs('../score/task 2/')
    SubmitGenerator(prediction, '../data/task2_sample_submission.csv', True, '../score/task 2/task2_submission.csv')


if __name__ == '__main__':
    main()
