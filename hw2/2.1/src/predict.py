from torch.utils.data import DataLoader
from tqdm import tqdm
import torch


def make_prediction(model, testData, word_dict, result_path, device):
    model.train(False)
    dataloader = DataLoader(dataset=testData,
                            batch_size=32,
                            shuffle=False,
                            collate_fn=testData.collate_fn,
                            num_workers=4)
    trange = tqdm(enumerate(dataloader), total=len(dataloader), desc='Predict')
    prediction = []
    for i, (x, data_len, datas_idx) in trange:
        o_labels = model(x.to(device), data_len=data_len, teacher_forcing_ratio=0.0, training=False)
        o_labels = torch.argmax(o_labels, dim=2)
        o_labels = o_labels.to('cpu').numpy().tolist()
        for idx in datas_idx:
            prediction.append(o_labels[idx])


def output(prediction, word_dict, result_path):
    with open(result_path, 'w', encoding="utf-8") as f:
        for idx, sent in enumerate(prediction):
            s = ' '
            seq = ["<SOS>"] + [word_dict.idx2word(word_idx) for word_idx in sent]
            f.writelines(s.join(seq) + '\n')
