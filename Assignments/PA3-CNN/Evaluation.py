import numpy as np
import torch


class Evaluation:

    # torch : k-hot encoding
    def __init__(self, predicts, targets):
        self.predicts = predicts
        self.targets = targets

        #   1     1 / 1 (True Positive)
        #   inf   1 / 0 (False Positive)
        #   nan   0 / 0 (True Negative)
        #   0     0 / 1 (False Negative)
        confusion_vector = predicts / targets

        self.TP = torch.sum(confusion_vector == 1, dim=0).double()
        self.FP = torch.sum(confusion_vector == float('inf'), dim=0).double()
        self.TN = torch.sum(torch.isnan(confusion_vector), dim=0).double()
        self.FN= torch.sum(confusion_vector == 0, dim=0).double()

        print(self.TP, self.FP, self.TN, self.FN)

    def accuracy(self):
        return torch.sum((self.predicts == self.targets).double(), dim=0) / self.predicts.shape[0]

    def precision(self):
        return self.TP / (self.TP + self.FP)

    def recall(self):
        return self.TP / (self.TP + self.FN)

    def confusion_matrix(self):
        '''
            pred | tar |   Action
              0  |  0  |   + (no disease, no disease)
              0  |  1  |   + (current, no disease presented)
              1  |  1  |   + (current, current)
              1  |  0  |   + all (current, other ground truth)
        '''
        num_classes = self.predicts.shape[1] + 1
        matrix = np.zeros(shape=(num_classes, num_classes))
        for predicts, targets in zip(self.predicts, self.targets):
            for cls, (pred, target) in enumerate(zip(predicts, targets)):
                if pred:
                    if target:
                        matrix[cls][cls] += 1
                    else:
                        for tgt_cls, tar in enumerate(targets):
                            if tar:
                                matrix[cls][tgt_cls] += 1
                else:
                    if target:
                        matrix[cls][-1] += 1
                    else:
                        matrix[-1][-1] += 1
        return matrix / self.predicts.shape[0]

    def evaluate(self):
        print(self.accuracy())
        print(self.precision())
        print(self.recall())
        print(self.confusion_matrix())


if __name__=="__main__":

    prediction = torch.tensor([
        [1, 0, 0., 1],
        [1, 1., 0., 1],
        [0, 1., 0., 1]
    ])

    target = torch.tensor([
        [1, 1, 0., 1],
        [1, 0, 1., 1],
        [1, 1., 0., 1]
    ])

    eva = Evaluation(prediction, target)
    eva.evaluate()