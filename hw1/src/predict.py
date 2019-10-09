import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
from network import BertForMultiLabelSequenceClassification
import pandas as pd
import os
import argparse


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--max_len', default=256, type=int)
    parser.add_argument('--cuda', default=1, type=int)
    parser.add_argument('--checkpoint', default=0, type=int)
    args = parser.parse_args()

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
