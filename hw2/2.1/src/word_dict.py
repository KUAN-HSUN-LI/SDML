class Word_dict():
    def __init__(self, dataset, max_len):
        self.word_dict = {}
        self.idx_dict = {}
        self.add("<PAD>")
        for i in range(1, max_len + 1):
            self.add(str(i))
        self.add("<UNK>")

        self.make_dict(dataset)

    def make_dict(self, dataset):
        for sent in dataset:
            for word in sent:
                self.add(word)

    def add(self, word):
        if word not in self.word_dict:
            self.idx_dict[self.get_len()] = word
            self.word_dict[word] = self.get_len()

    def word2idx(self, word):
        if word in self.word_dict:
            return self.word_dict[word]
        else:
            return self.word_dict["<UNK>"]

    def idx2word(self, idx):
        return self.idx_dict[idx]

    def get_len(self):
        return len(self.word_dict)
