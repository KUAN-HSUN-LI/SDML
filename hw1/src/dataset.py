from torch.utils.data import Dataset
import torch


class AbstractDataset(Dataset):
    def __init__(self, data, pad_idx, max_len=500):
        self.data = data
        self.pad_idx = pad_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, datas):
        # get max length in this batch
        max_sent = max([len(data['Abstract']) for data in datas])
        max_len = max([min(len(sentence), self.max_len) for data in datas for sentence in data['Abstract']])
        batch_abstract = []
        batch_label = []
        sent_len = []
        for data in datas:
            # padding abstract to make them in same length
            pad_abstract = []
            for sentence in data['Abstract']:
                if len(sentence) > max_len:
                    pad_abstract.append(sentence[:max_len])
                else:
                    pad_abstract.append(sentence + [self.pad_idx] * (max_len - len(sentence)))
            sent_len.append(len(pad_abstract))
            pad_abstract.extend([[self.pad_idx] * max_len] * (max_sent - len(pad_abstract)))
            batch_abstract.append(pad_abstract)

            # gather labels
            if 'Label' in data:
                batch_label.append(data['Label'])

        return torch.LongTensor(batch_abstract), torch.FloatTensor(batch_label), sent_len
