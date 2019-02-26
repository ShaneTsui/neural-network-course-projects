import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from utils.utils import file_preprocess

class TextDataset(Dataset):

    def  __init__(self, conf):
        self.filename = conf['filename']
        self.chunk_size = conf['chunk_size']
        self.voc_size = conf['voc_size']
        self.char2num = conf['char2num']
        self.num2char = conf['num2char']
        try:
            with open(self.filename, 'r') as f:
                self.text = f.read()
                self.chars = set(self.text)
                self.text_length = len(self.text)

                # Todo: check what to do to the very last characters
                self.len = (len(self.text) - 2) // self.chunk_size + 1
                # self.len = len(self.text)// self.chunk_size

        except KeyError as e:
            print("Key Error: {}".format(e))
        except Exception as e:
            print(e)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # Use characters in the very last via dynamically adjusted chunk_size
        left_boundary = idx * self.chunk_size
        right_boundary = min((idx + 1) * self.chunk_size + 1, self.text_length) # TODO: Note it shouldn't get out-of-boundary by 2. Not included

        txt = self.text[left_boundary:right_boundary]
        input_idx = torch.LongTensor([self.char2num[ch] for ch in txt[:-1]])
        input_idx.resize_((len(input_idx), 1))
        target_idx = torch.LongTensor([self.char2num[ch] for ch in txt[1:]])
        target_idx.resize_((len(target_idx), 1))

        return self._encode(input_idx, target_idx)

    # input and target are 2 tensors of char index
    def _encode(self, input_idx, target_idx):
        assert input_idx.shape == target_idx.shape
        chunk_size = len(input_idx)
        input = torch.zeros(chunk_size, self.voc_size)
        target = torch.zeros(chunk_size, self.voc_size)
        # for i in range(chunk_size):
        #     input[i][input_idx[i]] = 1.
        #     target[i][target_idx[i]] = 1.
        input.scatter_(1, input_idx, 1.)
        target.scatter_(1, target_idx, 1.)

        return input, target

    def _one_hot_encode(self, idx):
        vector = torch.zeros(self.voc_size).long()
        vector[idx] = 1
        return vector

    @property
    def vocabulary_size(self):
        return self.voc_size

def split_dataset(filename):

    conf = yaml.load(open(filename, encoding='utf-8'))
    conf_train = conf['train']
    conf_val= conf['val']
    conf_test= conf['test']

    voc_size, char2num, num2char = file_preprocess(conf_train['filename'])
    conf_train['voc_size'], conf_train['char2num'], conf_train['num2char'] = voc_size, char2num, num2char
    conf_val['voc_size'], conf_val['char2num'], conf_val['num2char'] = voc_size, char2num, num2char
    conf_test['voc_size'], conf_test['char2num'], conf_test['num2char'] = voc_size, char2num, num2char

    dataset_train = TextDataset(conf_train)
    dataset_val = TextDataset(conf_val)
    dataset_test = TextDataset(conf_test)

    dataloader_train = DataLoader(dataset_train, batch_size=1, num_workers=0, pin_memory=True)
    dataloader_val = DataLoader(dataset_val, batch_size=1, num_workers=0, pin_memory=True)
    dataloader_test = DataLoader(dataset_test, batch_size=1, num_workers=0, pin_memory=True)

    return dataloader_train, dataloader_val, dataloader_test, conf