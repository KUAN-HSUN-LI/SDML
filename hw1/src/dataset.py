from torch.utils.data import Dataset
import torch


class BertDataset(Dataset):
    def __init__(self, data, max_len):
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, datas):
        # get max length in this batch
        max_len = max([min(len(data['tokens']), self.max_len) for data in datas])
        batch_tokens = []
        batch_segments = []
        batch_masks = []
        batch_label = []
        for data in datas:
            # padding abstract to make them in same length
            abstract_len = len(data['tokens'])
            if abstract_len > max_len:
                batch_tokens.append(data['tokens'][:max_len])
                # batch_segments.append(data['segments'][:max_len])
                batch_segments.append([0] * max_len)
                batch_masks.append([1] * max_len)
            else:
                batch_tokens.append(data['tokens'] + [0] * (max_len - abstract_len))
                # batch_segments.append(data['segments'] + [0] * (max_len - abstract_len))
                batch_segments.append([0] * max_len)
                batch_masks.append([1] * abstract_len + [0] * (max_len - abstract_len))
            # gather labels
            if 'Label' in data:
                batch_label.append(data['Label'])
        return torch.LongTensor(batch_tokens), torch.LongTensor(batch_segments), torch.LongTensor(
            batch_masks), torch.FloatTensor(batch_label)
