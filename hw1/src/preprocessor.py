import pandas as pd
from multiprocessing import Pool
from transformers import BertTokenizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import ipdb
import torch


class Preprocessor:

    def __init__(self, pretrained_model_name, embedding):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.embedding = embedding

    def label_to_onehot(self, labels):
        """ Convert label to onehot .
            Args:
                labels (string): sentence's labels.
            Return:
                outputs (onehot list): sentence's onehot label.
        """
        label_dict = {'THEORETICAL': 0, 'ENGINEERING': 1, 'EMPIRICAL': 2}
        onehot = [0, 0, 0]
        for l in labels.split():
            if l == "OTHERS":
                continue
            onehot[label_dict[l]] = 1
        return onehot

    def sentence_to_indices(self, sentence):
        """ Convert sentence to its word indices.
        Args:
            sentence (str): One string.
        Return:
            indices (list of int): List of word indices.
        """
        return [self.tokenizer.convert_tokens_to_ids(word) for word in self.tokenizer.tokenize(sentence)]

    def get_dataset(self, dataset, n_workers=4):

        results = [None] * n_workers
        tfidf_results = [None] * n_workers
        with Pool(processes=n_workers) as pool:
            for i in range(n_workers):
                batch_start = (len(dataset) // n_workers) * i
                if i == n_workers - 1:
                    batch_end = len(dataset)
                else:
                    batch_end = (len(dataset) // n_workers) * (i + 1)

                batch = dataset[batch_start: batch_end]
                results[i] = pool.apply_async(self.preprocess_samples, [batch])
                tfidf_results[i] = pool.apply_async(self.preprocess_tfidf_samples, [batch])

            pool.close()
            pool.join()

        processed = []
        for result in results:
            processed += result.get()
        tfidf_processed = []
        for tfidf_result in tfidf_results:
            tfidf_processed += tfidf_result.get()

        # processed = self.preprocess_samples(dataset)
        # tfidf_processed = self.preprocess_tfidf_samples(dataset)

        return processed, tfidf_processed

    def preprocess_samples(self, dataset):
        """ Worker function.

        Args:
            dataset (list of dict)
        Returns:
            list of processed dict.
        """
        processed = []
        for sample in tqdm(dataset.iterrows(), total=len(dataset)):
            processed.append(self.preprocess_sample(sample[1]))

        return processed

    def preprocess_sample(self, data):
        """
        Args:
            data (dict)
        Returns:
            dict
        """
        processed = dict()
        processed['tokens'] = [self.sentence_to_indices(sent) for sent in data['Abstract'].split('$$$')]
        processed['tokens'] = sum(processed['tokens'], [])
        processed['tokens'] = [self.tokenizer.convert_tokens_to_ids('[CLS]')] + processed['tokens'] + [
            self.tokenizer.convert_tokens_to_ids('[SEP]')]
        # processed['segments'] = [0] * len(processed['tokens'])

        if 'Task 2' in data:
            processed['Label'] = self.label_to_onehot(data['Task 2'])

        return processed

    def preprocess_tfidf_samples(self, dataset):
        """ Worker function.

        Args:
            dataset (list of dict)
        Returns:
            list of processed dict.
        """
        processed = []
        for sample in tqdm(dataset.iterrows(), total=len(dataset)):
            processed.append(self.preprocess_tfidf_sample(sample[1]))

        return processed

    def preprocess_tfidf_sample(self, data):

        processed = [self.embedding.to_index(word)
                     for sent in data['Abstract'].split('$$$') for word in word_tokenize(sent)]
        # processed = sum(processed, [])
        return processed

    def get_glove_tfidf_emb(self, dataset, tfidf):
        data_num = 0
        abstract_embs = torch.Tensor()
        for data in tqdm(dataset.iterrows(), total=len(dataset)):
            abstract_emb = torch.zeros(300)
            word_num = 0
            for sentence in data[1]['Abstract'].split('$$$'):
                for word in word_tokenize(sentence):
                    word_idx = self.embedding.to_index(word)
                    word_emb = self.embedding.vectors[word_idx]
                    word_emb *= tfidf[data_num][word_num]      # (tensor[300])
                    abstract_emb += word_emb
                    word_num += 1
            # abstract_emb /= word_num
            # abstract_emb *= 10
            abstract_embs = torch.cat((abstract_embs, abstract_emb.unsqueeze(0)))
            data_num += 1

        return abstract_embs        # tensor[6300, 300]
