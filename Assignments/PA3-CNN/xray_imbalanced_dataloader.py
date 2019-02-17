import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from torchvision import transforms
from PIL import Image

import numpy as np

from xray_dataloader import ChestXrayDataset


class ImbalancedDatasetSampler(Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            labels = dataset.get_labels(idx)
            for label in labels:
                if label in label_to_count:
                    label_to_count[label] += 1
                else:
                    label_to_count[label] = 1
        print(label_to_count)

        # weight for each sample
        weights = [1.0 / min([label_to_count[label] for label in dataset.get_labels(idx)]) for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)
        print(self.weights)

    def __iter__(self):
        # https://pytorch.org/docs/stable/torch.html?highlight=torch%20multinomial#torch.multinomial
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


def create_balanced_split_loaders(batch_size, seed, transform=transforms.ToTensor(),
                         p_val=0.1, p_test=0.2, shuffle=True,
                         show_sample=False, extras={}, z_score=False):
    """ Creates the DataLoader objects for the training, validation, and test sets.

    Params:
    -------
    - batch_size: (int) mini-batch size to load at a time
    - seed: (int) Seed for random generator (use for testing/reproducibility)
    - transform: A torchvision.transforms object - transformations to apply to each image
                 (Can be "transforms.Compose([transforms])")
    - p_val: (float) Percent (as decimal) of dataset to use for validation
    - p_test: (float) Percent (as decimal) of the dataset to split for testing
    - shuffle: (bool) Indicate whether to shuffle the dataset before splitting
    - show_sample: (bool) Plot a mini-example as a grid of the dataset
    - extras: (dict)
        If CUDA/GPU computing is supported, contains:
        - num_workers: (int) Number of subprocesses to use while loading the dataset
        - pin_memory: (bool) For use with CUDA - copy tensors into pinned memory
                  (set to True if using a GPU)
        Otherwise, extras is an empty dict.

    Returns:
    --------
    - train_loader: (DataLoader) The iterator for the training set
    - val_loader: (DataLoader) The iterator for the validation set
    - test_loader: (DataLoader) The iterator for the test set
    """

    augmentation = transforms.Compose([transforms.RandomRotation(20, resample=Image.BILINEAR),
                                       transforms.CenterCrop(900),
                                       transforms.Resize(512),
                                       transforms.ToTensor()])


    # Get create a ChestXrayDataset object
    dataset = ChestXrayDataset(transform, z_score=z_score)
    dataset_train = ChestXrayDataset(augmentation, z_score=z_score)

    # Dimensions and indices of training set
    dataset_size = len(dataset)
    all_indices = list(range(dataset_size))

    # Shuffle dataset before dividing into training & test sets
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(all_indices)

    # Create the validation split from the full dataset
    val_split = int(np.floor(p_val * dataset_size))
    train_ind, val_ind = all_indices[val_split:], all_indices[: val_split]

    # Separate a test split from the training dataset
    test_split = int(np.floor(p_test * len(train_ind)))
    train_ind, test_ind = train_ind[test_split:], train_ind[: test_split]

    print("start creating imbalanced sampler")
    sample_train = ImbalancedDatasetSampler(dataset_train, indices=train_ind)
    sample_test = SubsetRandomSampler(test_ind)
    sample_val = SubsetRandomSampler(val_ind)
    print("end creating imbalanced sampler")

    num_workers = 1
    pin_memory = True
    # If CUDA is available
    #if extras:
    #    num_workers = extras["num_workers"]
    #    pin_memory = extras["pin_memory"]

    # Define the training, test, & validation DataLoaders

    train_loader = DataLoader(dataset_train, batch_size=batch_size,
                              sampler=sample_train, num_workers=4,
                              pin_memory=pin_memory)
    print("train_loader created")

    test_loader = DataLoader(dataset, batch_size=batch_size,
                             sampler=sample_test, num_workers=4,
                             pin_memory=pin_memory)
    print("test_loader created")

    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=sample_val, num_workers=4,
                            pin_memory=pin_memory)
    print("val_loader created")

    print("start calculating label weights")
    label_weights = dataset.get_weights()
    print("weights:", label_weights)

    # Return the training, validation, test DataLoader objects
    return (train_loader, val_loader, test_loader, label_weights)