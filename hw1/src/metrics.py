import torch


class F1:
    def __init__(self):
        self.threshold = 0.0
        self.n_precision = 0
        self.n_recall = 0
        self.n_corrects = 0
        self.name = 'F1'

    def reset(self):
        self.n_precision = 0
        self.n_recall = 0
        self.n_corrects = 0

    def update(self, predicts, groundTruth):
        predicts = predicts > self.threshold
        predicts = predicts.type(torch.float32)
        predicts = self.add_labels(predicts)
        groundTruth = self.add_labels(groundTruth)
        self.n_precision += torch.sum(predicts).data.item()
        self.n_recall += torch.sum(groundTruth).data.item()
        self.n_corrects += torch.sum(groundTruth * predicts).data.item()

    def add_labels(self, labels):
        new_axis = []
        for label in labels:
            new_axis.append([1]) if torch.sum(label) == 0 else new_axis.append([0])
        return torch.cat((labels, torch.Tensor(new_axis)), 1)

    def get_score(self):
        recall = self.n_corrects / self.n_recall
        precision = self.n_corrects / (self.n_precision + 1e-20)
        return 2 * (recall * precision) / (recall + precision + 1e-20)

    def print_score(self):
        score = self.get_score()
        return '{:.5f}'.format(score)
