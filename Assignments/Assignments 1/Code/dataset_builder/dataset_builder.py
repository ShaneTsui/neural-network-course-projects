from dataset_builder.subject import Subject
from dataset_builder.dataset import Dataset

import numpy as np
import random
from os import listdir
from PIL import Image
from collections import defaultdict

class DatasetBuilder:
    def __init__(self):
        self.subjects = defaultdict(Subject)

    def get_subject_ids(self):
        return list(self.subjects.keys())

    def load_data(self, data_dir="./data/CAFE/"):
        # Get the list of image file names
        all_files = listdir(data_dir)

        # Store the images and labels in self.subjects dictionary
        for file in all_files:
            # Load in the files as PIL images and convert to NumPy arrays
            subject, rest_string = file.split('_')
            label = rest_string.split('.')[0][:-1]

            # Exclude neutral and happy faces
            if label != 'n' and label != 'h':
                img = Image.open(data_dir + file)
                self.subjects[subject].add(np.array(img, dtype=np.float64).reshape(-1, ),
                                           label)  # Reshaped to a vector

    def build_dataset(self, test_subject_id, labels):
        train, holdout, test, pca = Dataset(), Dataset(), Dataset(), []

        # Select data for train, holdout and test dataset
        subject_ids = self.get_subject_ids()
        test_subject = self.subjects[test_subject_id]
        subject_ids.remove(test_subject_id)

        holdout_subject_id = random.choice(subject_ids)
        holdout_subject = self.subjects[holdout_subject_id]
        subject_ids.remove(holdout_subject_id)

        for label in labels:
            test.insert(test_subject.get(label), label)
            holdout.insert(holdout_subject.get(label), label)
            train.extend([self.subjects[train_subject_id].get(label) for train_subject_id in subject_ids],
                         [label] * len(subject_ids))

        # Select data for PCA
        for train_subject_id in subject_ids:
            pca.extend(list(self.subjects[train_subject_id].label_image_dict.values()))

        # To numpy array
        train.to_numpy_array()
        holdout.to_numpy_array()
        test.to_numpy_array()
        pca = np.array(pca)

        # Normalizatiton
        mean = np.mean(pca, axis=0)
        pca_normalized = (pca - mean)
        train.data = (train.data - mean)
        holdout.data = (holdout.data - mean)
        test.data = (test.data - mean)

        return train, holdout, test, pca_normalized