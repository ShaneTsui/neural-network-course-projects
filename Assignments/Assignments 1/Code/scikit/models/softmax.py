from scikit.models.classifier import Classifier
import numpy as np

class SoftmaxClassifier(Classifier):
    def encoder(self, labels):
        self.label_set = list(set(labels))
        self.one_hot_dict = {label: one_hot for label, one_hot in zip(self.label_set, np.eye(len(self.label_set)))}
        return lambda labels: np.array([self.one_hot_dict[label] for label in labels])

    def softmax(self, s):
        return np.exp(s) / np.sum(np.exp(s), axis=1, keepdims=True)

    def predict(self, X):
        probs = self.softmax(np.dot(X, self.w))
        return np.argmax(probs, axis=1)[:, np.newaxis]

    def loss(self, X, y):
        y_hat = self.softmax(np.dot(X, self.w))
        return - np.sum(y * np.log(y_hat)) / len(y_hat)

    def gradient(self, X, y):
        y_hat = self.softmax(np.dot(X, self.w))
        return np.dot(X.T, (y_hat - y))

    def accuracy(self, test_set=None):
        if not test_set:
            test_set = self.test_set
        line, self.test_set.y = np.where(test_set.y == 1.)
        return np.sum(self.predict(test_set.X).T[0] == test_set.y) / len(test_set.y)