from dataset_builder.dataset_builder import DatasetBuilder
from scikit.decomposition.pca import PCA
from scikit.models.logistic import LogisticClassifier
from scikit.models.softmax import SoftmaxClassifier
from utils.recorder import Records
from utils.display.display import *

import numpy as np

class Pipeline:
    def __init__(self, facial_expressions, dataset_dir='./data/CAFE/', classifier_type="logistic", pca=True):
        self.facial_expressions = facial_expressions

        self.data_builder = DatasetBuilder()
        self.data_builder.load_data(dataset_dir)

        if classifier_type == "logistic":
            self.classifier = LogisticClassifier()
        elif classifier_type == "softmax":
            self.classifier = SoftmaxClassifier()

        if pca:
            self.pca = PCA()

        self.records = Records(n_labels=len(facial_expressions))

    def build(self, n_components, learning_rate, n_epoches, batch_size=None, n_repeats=10):
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.n_epoches = n_epoches
        self.batch_size = batch_size
        self.n_repeats = n_repeats

    def run(self):

        subjects = self.data_builder.get_subject_ids()
        for repeat in range(self.n_repeats):
            train_losses, holdout_losses, test_accuracies = [], [], []
            for test_subject in subjects:
                # Dataset building
                train, holdout, test, pca_data = self.data_builder.build_dataset(test_subject, self.facial_expressions)
                self.pca.learn(pca_data)

                # from utils.display.display import display_eigs
                # display_eigs(self.pca.eig_vecs.T[::-1].T)
                # print("Done!!!")
                # break

                train.data = self.pca.run(train.data, self.n_components)
                holdout.data = self.pca.run(holdout.data, self.n_components)
                test.data = self.pca.run(test.data, self.n_components)

                # Load data
                self.classifier.load_data(train, holdout, test)

                # Train
                self.classifier.train(lr=self.learning_rate, T=self.n_epoches, bs=self.batch_size)

                # Test
                self.classifier.test(confusion=True)

                # Recording results
                train_losses.append(self.classifier.train_losses)
                holdout_losses.append(self.classifier.holdout_losses)
                test_accuracies.append(self.classifier.test_accuracies)
                self.records.test_confusion += self.classifier.confusion_matrix
            # print(self.classifier.test_accuracies)
            self.records.record(np.array(train_losses).mean(axis=0).tolist(),
                                np.array(holdout_losses).mean(axis=0).tolist(),
                                np.array(test_accuracies).mean().tolist())
        self.records.test_confusion = self.records.test_confusion / np.sum(self.records.test_confusion, axis=1)

    def display_faces(self, facial_expressions=['ht', 's', 'a', 'f', 'm', 'd']):
        subjects = self.data_builder.get_subject_ids()
        test_subject = subjects[0]
        train, _, _, _= self.data_builder.build_dataset(test_subject, facial_expressions)
        display_six_expressions(train.data, train.labels)

    def display_eigenface(self):
        display_eigs(self.pca.get_eig_vecs())


    def visualize_weights(self):
        print(self.classifier.label_set)
        visualized_weights = visualize_weights(self.classifier.w.T, self.pca.eig_vecs, self.classifier.label_set)
        print(visualized_weights)