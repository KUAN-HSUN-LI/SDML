from torch.utils.data import Dataset
import torch


class BertDataset(Dataset):
    def __init__(self, data, node_vec, tfidf, max_len):
        self.data = data
        self.max_len = max_len
        self.node_vec = node_vec
        self.tfidf = tfidf

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            'bert_data': self.data[index],
            'node_vec': self.node_vec[index],
            'tfidf': self.tfidf[index]
        }

    def collate_fn(self, datas):
        # get max length in this batch
        max_len = max([min(len(data['bert_data']['tokens']), self.max_len) for data in datas])
        batch_tokens = []
        batch_segments = []
        batch_masks = []
        batch_label = []
        batch_node_vec = []
        batch_tfidf = []
        for data in datas:
            batch_node_vec.append(data['node_vec'])
            # padding abstract to make them in same length
            abstract_len = len(data['bert_data']['tokens'])
            if abstract_len > max_len:
                batch_tokens.append(data['bert_data']['tokens'][:max_len])
                batch_segments.append([0] * max_len)
                batch_masks.append([1] * max_len)
                batch_tfidf.append(data['tfidf'][:max_len])
            else:
                batch_tokens.append(data['bert_data']['tokens'] + [0] * (max_len - abstract_len))
                batch_segments.append([0] * max_len)
                batch_masks.append([1] * abstract_len + [0] * (max_len - abstract_len))
                batch_tfidf.append(data['tfidf'] + [0.0] * (max_len - abstract_len))
            # gather labels
            if 'Label' in data['bert_data']:
                batch_label.append(data['bert_data']['Label'])
        batch_node_vec = torch.stack(batch_node_vec)
        return torch.LongTensor(batch_tokens), torch.LongTensor(batch_segments), torch.LongTensor(
            batch_masks), batch_node_vec, torch.FloatTensor(batch_tfidf), torch.FloatTensor(batch_label)
