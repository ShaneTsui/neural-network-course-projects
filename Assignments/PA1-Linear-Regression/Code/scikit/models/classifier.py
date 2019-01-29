import numpy as np

class Classifier:
    def __init__(self):
        self.train_set, self.holdout_set, self.test_set = None, None, None
        self.w = None

    def load_data(self, train, holdout, test):
        self.train_set, self.holdout_set, self.test_set = train, holdout, test

        self.encode = self.encoder(train.labels)

        for dataset in [self.train_set, self.holdout_set, self.test_set]:
            dataset.X = self.bias(dataset.data)
            dataset.y = self.encode(dataset.labels)

    def bias(self, data):
        return np.column_stack((np.ones(len(data)), data))

    def test(self, confusion=False):
        self.test_accuracies.append(self.accuracy())
        if confusion:
            self.confusion()

    def train(self, T=10, lr=0.06, bs=None):

        # initialization
        train_X, train_y = self.train_set.X, self.train_set.y
        holdout_X, holdout_y = self.holdout_set.X, self.holdout_set.y
        self.train_losses, self.holdout_losses = [], []
        self.w = np.random.random((train_X.shape[1], train_y.shape[1]))
        self.W = []
        self.test_accuracies = []

        # gradient descent
        for t in range(T):
            # gradient descent on each batch
            if bs:
                # generate random permutation
                perm = np.random.permutation(len(train_X))
                for i in range(round(len(train_X) / bs)):
                    train_X_batch, train_y_batch = train_X[perm[i:i + bs]], train_y[perm[i:i + bs]]
                    self.w -= lr * self.gradient(train_X_batch, train_y_batch)
            else:
                self.w -= lr * self.gradient(train_X, train_y)
            self.W.append(self.w.tolist())

            # compute losses on train dataset and holdout dataset
            self.train_losses.append(self.loss(train_X, train_y))
            self.holdout_losses.append(self.loss(holdout_X, holdout_y))

            # save the parameters with best performance
        #         print("The W: {}".format(np.array(self.W)))
        self.w = np.array(min(self.W, key=lambda w: self.holdout_losses[self.W.index(w)]))

    def confusion(self, test_set=None):
        if not test_set:
            test_set = self.test_set
        X, y = test_set.X, test_set.y
        self.confusion_matrix = np.zeros((X.shape[0], X.shape[0]))
        prediction = self.predict(X).T[0]
        for p, label in zip(prediction, y):
            self.confusion_matrix[label][p] += 1