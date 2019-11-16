import torch


def load_pkl(pkl_path):
    import pickle
    with open(pkl_path, mode='rb') as f:
        obj = pickle.load(f)

    return obj


class accuracy():
    def __init__(self):
        self.ncontrol_word = 0
        self.nword_correct = 0

    def update(self, predictions, groundTruths, random_nums):
        predictions = predictions.cpu()
        groundTruths = groundTruths.cpu()
        for i in range(len(groundTruths)):
            prediction = predictions[i]
            groundTruth = groundTruths[i]
            random_num = random_nums[i]
            self.ncontrol_word += len(random_num)
            for num in random_num:
                if prediction[num-1] == groundTruth[num-1]:
                    self.nword_correct += 1

    def get_word_score(self):
        return self.nword_correct / self.ncontrol_word

    def print_word_score(self):
        score = self.get_word_score()
        return '{:.5f}'.format(score)
