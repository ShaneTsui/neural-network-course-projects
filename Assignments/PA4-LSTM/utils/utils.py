import torch

def encode(character, char2num):
    voc_size = len(char2num)
    one_hot_encode = torch.zeros(voc_size)
    one_hot_encode[char2num[character]] = 1.
    return one_hot_encode

def one_hot_encode(idx, voc_size):
    vector = torch.zeros(voc_size)
    vector[idx] = 1.
    return vector

def file_preprocess(filename):
    with open(filename) as f:
        text = f.read()
        chars = set(text)
        voc_size = len(chars)
        char2num = {ch: idx for idx, ch in enumerate(sorted(chars))}
        num2char = {idx: ch for idx, ch in enumerate(sorted(chars))}
        return voc_size, char2num, num2char