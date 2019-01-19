from matplotlib import pyplot as plt
import numpy as np

class Records:
    def __init__(self, n_labels):
        #         self.train_losses = np.array([])
        #         self.holdout_losses = np.array([])
        #         self.test_accuracies = np.array([])

        self.train_losses = []
        self.holdout_losses = []
        self.test_accuracies = []
        self.test_confusion = np.zeros((n_labels, n_labels))

    def record(self, train_los, holdout_los, test_acc):

        #         np.append(self.train_losses, train_los)
        #         np.append(self.holdout_losses, holdout_los)
        #         np.append(self.test_accuracies, test_acc)

        self.train_losses.append(train_los)
        self.holdout_losses.append(holdout_los)
        self.test_accuracies.append(test_acc)

    def plt_losses(self, n_components, lr, n_epoches, train=True, holdout=True):
        assert len(self.train_losses) == len(self.holdout_losses)
        plt.figure()
        if train:
            plt.errorbar(range(n_epoches), np.mean(self.train_losses, axis=0), yerr=np.std(self.train_losses, axis=0))
        if holdout:
            plt.errorbar(range(n_epoches), np.mean(self.holdout_losses, axis=0),
                         yerr=np.std(self.holdout_losses, axis=0))
        plt.title("n_components={}, learning_rates={}, n_epoches={}".format(n_components, lr, n_epoches))
        plt.show()

    def show_accuracies(self):
        print("The accuracy is {}".format(np.mean(self.test_accuracies)))

    def plt_confusion(self):
        plt.matshow(self.test_confusion)