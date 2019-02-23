import torch
import torch.nn as nn

class Transfer:
    def __init__(self, n_class, finetuning=False):
        self.finetuning = finetuning
        self.n_class = n_class

    def __call__(self, model):
        if not self.finetuning:
            self.gradFreeze(model)
        self.fcRest(model)
        return model

    def fcRest(self, pretrained):
        num_ins = pretrained.fc.in_features
        pretrained.fc = nn.Linear(num_ins, self.n_class)

    def gradFreeze(self, pretrained):
        for param in pretrained.parameters():
            param.requires_grad = False