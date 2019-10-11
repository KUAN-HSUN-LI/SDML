from torch.utils.data import Dataset
import torch


class BertDataset(Dataset):
    def __init__(self, data, tfidf_data, max_len):
        self.data = data
        self.tfidf_data = tfidf_data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            'bert_data': self.data[index],
            'tfidf_data': self.tfidf_data[index] + [0] * 2
        }

    def collate_fn(self, datas):
        # get max length in this batch
        max_len = max([min(len(data['bert_data']['tokens']), self.max_len) for data in datas])
        batch_tokens = []
        batch_segments = []
        batch_masks = []
        batch_label = []
        batch_tfidf = []
        for data in datas:
            bert_data = data['bert_data']
            tfidf_data = data['tfidf_data']

            # padding abstract to make them in same length
            abstract_len = len(bert_data['tokens'])
            if abstract_len > max_len:
                batch_tokens.append(bert_data['tokens'][:max_len])
                # batch_segments.append(bert_data['segments'][:max_len])
                batch_segments.append([0] * max_len)
                batch_masks.append([1] * max_len)
                batch_tfidf.append(tfidf_data[:self.max_len])
            else:
                batch_tokens.append(bert_data['tokens'] + [0] * (max_len - abstract_len))
                # batch_segments.append(bert_data['segments'] + [0] * (max_len - abstract_len))
                batch_segments.append([0] * max_len)
                batch_masks.append([1] * abstract_len + [0] * (max_len - abstract_len))
                batch_tfidf.append(tfidf_data + [0.0] * (self.max_len - abstract_len))
            # gather labels
            if 'Label' in bert_data:
                batch_label.append(bert_data['Label'])
        return torch.LongTensor(batch_tokens), torch.LongTensor(batch_segments), torch.LongTensor(
            batch_masks), torch.FloatTensor(batch_tfidf), torch.FloatTensor(batch_label)
