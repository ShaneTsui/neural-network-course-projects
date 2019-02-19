import torch.nn as nn
import torch
import torch.nn.functional as F

class weighted_loss(nn.Module):
    def __init__(self, weight=None, Verbose=False):
        super(weighted_loss, self).__init__()
        self.weight = weight
        self.Verbose = Verbose

    def forward(self, input, target):
        class_sum = torch.sum(target, dim=0)
        positive_ratio = class_sum / target.size()[0]  # P / (P+N)
        negative_ratio = torch.ones(positive_ratio.size()).cuda() - positive_ratio  # N / (P+N)

        loss = -(target * F.logsigmoid(input) * negative_ratio + (1 - target) * F.logsigmoid(-input) * positive_ratio)

        loss = loss.sum(dim=1) / input.size(1)
        ret = loss.mean()

        return ret

class w_cel_loss(nn.Module):
    def __init__(self, weight=None):
        super(w_cel_loss, self).__init__()
        self.weight = weight

    def forward(self, input, target):
        P = torch.sum(target)
        N = target.shape[0] * target.shape[1] - P

        beta_p = (P + N) / P
        beta_n = (P + N) / N
        loss = torch.mean(-(beta_p * target * F.logsigmoid(input) + beta_n * (1 - target) * F.logsigmoid(-input)))
        return loss
