import numpy as np
from utils.display.display import *

class Dataset:
    def __init__(self, data=[], labels=[]):
        self.data = []
        self.labels = []

    def to_numpy_array(self):
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

    def insert(self, datum, label):
        self.data.append(datum)
        self.labels.append(label)

    def extend(self, data, labels):
        self.data.extend(data)
        self.labels.extend(labels)

    def shuffle(self):
        idx = np.array(list(range(len(self.data))))
        np.random.shuffle(idx)
        self.data[:] = self.data[idx]
        self.labels[:] = self.labels[idx]