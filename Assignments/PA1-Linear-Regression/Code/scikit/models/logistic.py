from scikit.models.classifier import Classifier
import numpy as np

class LogisticClassifier(Classifier):
    def encoder(self, labels):
        label_set = list(set(labels))
        assert len(label_set) == 2
        binary_dict = {label_set[0]: 0, label_set[1]: 1}
        return lambda labels: np.array([binary_dict[label] for label in labels]).reshape(-1, 1)

    def logistic(self, s):
        return np.array(1 / (1 + np.exp(-s)))

    def predict(self, X):
        probs = self.logistic(np.dot(X, self.w))
        prediction = np.zeros(probs.shape)
        prediction[probs > 0.5] = 1
        return prediction

    def loss(self, X, y):
        y_hat = self.logistic(np.dot(X, self.w))
        return (- np.dot(y.T, np.log(y_hat)) / len(y_hat))[0][0]

    def gradient(self, X, y):
        y_hat = self.logistic(np.dot(X, self.w))
        return np.sum((y_hat - y) * X, axis=0).reshape(-1, 1)

    def accuracy(self, test_set=None):
        if not test_set:
            test_set = self.test_set
        return np.sum(self.predict(test_set.X) == test_set.y) / len(test_set.y)