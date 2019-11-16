import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset
import random
random.seed(42)


class SentenceDataset(Dataset):
    def __init__(self, data, word_dict, pad_idx, max_len=25, ncontrols=1, training=True):
        self.data = data
        self.word_dict = word_dict
        self.pad_idx = pad_idx
        self.max_len = max_len
        self.training = training
        self.ncontrols = ncontrols

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, datas):
        if self.training:
            processed_data = []
            for data in datas:
                processed = {}
                processed['x'] = data['x'].copy()
                processed['y'] = data['y']
                processed_data.append(self.process_data(processed))

            processed_data.sort(key=lambda x: len(x['x']), reverse=True)
            batch_data = [data['x'] for data in processed_data]

            datas_len = [len(data) for data in batch_data]
            batch_data = self.make_padding(batch_data)

            batch_label = [data['y'] for data in processed_data]
            batch_label = self.make_padding(batch_label)
            batch_random_num = [data['random_num'] for data in processed_data]

            return torch.LongTensor(batch_data), torch.LongTensor(batch_label), torch.LongTensor(datas_len), batch_random_num

        else:
            datas_idx = sorted(range(len(datas)), key=lambda x: datas[x])
            datas.sort(key=lambda x: len(x), reverse=True)
            datas_len = [len(sent) for sent in datas]
            batch_data = self.make_padding(datas)
            return torch.LongTensor(batch_data), torch.LongTensor(datas_len), datas_idx

    def process_data(self, data):
        data['random_num'] = self.get_ncontrols(len(data['y'])-2)
        data['x'] = self.add_control(data['x'], data['y'], data['random_num'])
        return data

    def get_ncontrols(self, sent_len):
        sample_num = random.randint(1, self.ncontrols)
        if sent_len < sample_num:
            sample_num = sent_len

        samples = random.sample([i for i in range(1, sent_len+1)], sample_num)
        samples.sort()
        return samples

    def add_control(self, data, label, samples):
        for sample in samples:
            data = data + [self.word_dict.word2idx(str(sample))] + [label[sample]]
        return data

    def make_padding(self, datas):
        max_len = max([len(data) for data in datas])
        return [data + [self.pad_idx]*(max_len-len(data)) for data in datas]
