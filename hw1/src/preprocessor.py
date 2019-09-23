import pandas as pd
from multiprocessing import Pool
from nltk.tokenize import word_tokenize
# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm

from dataset import AbstractDataset


class Preprocessor:

    def collect_words(self, data_path, n_workers=4):
        df = pd.read_csv(data_path, dtype=str)

        sent_list = []
        for i in df.iterrows():
            sent_list += i[1]['Abstract'].split('$$$')

        chunks = [
            ' '.join(sent_list[i:i + len(sent_list) // n_workers])
            for i in range(0, len(sent_list), len(sent_list) // n_workers)
        ]
        with Pool(n_workers) as pool:
            chunks = pool.map_async(word_tokenize, chunks)
            words = set(sum(chunks.get(), []))

        return words

    def label_to_onehot(self, labels):
        """ Convert label to onehot .
            Args:
                labels (string): sentence's labels.
            Return:
                outputs (onehot list): sentence's onehot label.
        """
        label_dict = {'THEORETICAL': 0, 'ENGINEERING': 1, 'EMPIRICAL': 2, 'OTHERS': 3}
        onehot = [0, 0, 0, 0]
        for l in labels.split():
            onehot[label_dict[l]] = 1
        return onehot

    def sentence_to_indices(self, sentence, word_dict):
        """ Convert sentence to its word indices.
        Args:
            sentence (str): One string.
        Return:
            indices (list of int): List of word indices.
        """
        return [word_dict.to_index(word) for word in word_tokenize(sentence)]

    def get_dataset(self, data_path, word_dict, pad_idx=0, n_workers=4):
        """ Load data and return dataset for training and validating.

        Args:
            data_path (str): Path to the data.
        """
        dataset = pd.read_csv(data_path, dtype=str)

        results = [None] * n_workers
        with Pool(processes=n_workers) as pool:
            for i in range(n_workers):
                batch_start = (len(dataset) // n_workers) * i
                if i == n_workers - 1:
                    batch_end = len(dataset)
                else:
                    batch_end = (len(dataset) // n_workers) * (i + 1)

                batch = dataset[batch_start: batch_end]
                results[i] = pool.apply_async(self.preprocess_samples, args=(batch, word_dict))

            pool.close()
            pool.join()

        processed = []
        for result in results:
            processed += result.get()
        return AbstractDataset(processed, pad_idx)

    def preprocess_samples(self, dataset, word_dict):
        """ Worker function.

        Args:
            dataset (list of dict)
        Returns:
            list of processed dict.
        """
        processed = []
        for sample in tqdm(dataset.iterrows(), total=len(dataset)):
            processed.append(self.preprocess_sample(sample[1], word_dict))

        return processed

    def preprocess_sample(self, data, word_dict):
        """
        Args:
            data (dict)
        Returns:
            dict
        """
        processed = {}
        processed['Abstract'] = [self.sentence_to_indices(sent, word_dict)
                                 for sent in data['Abstract'].split('$$$')]
        if 'Task 2' in data:
            processed['Label'] = self.label_to_onehot(data['Task 2'])

        return processed

        return words
