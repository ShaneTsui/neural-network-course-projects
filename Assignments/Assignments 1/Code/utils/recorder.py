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

    # def plt_losses(self, n_components, lr, n_epoches, train=True, holdout=True):
    #     assert len(self.train_losses) == len(self.holdout_losses)
    #     plt.figure()
    #     if train:
    #         plt.errorbar(range(n_epoches), np.mean(self.train_losses, axis=0), yerr=np.std(self.train_losses, axis=0))
    #     if holdout:
    #         plt.errorbar(range(n_epoches), np.mean(self.holdout_losses, axis=0),
    #                      yerr=np.std(self.holdout_losses, axis=0))
    #     plt.title("n_components={}, learning_rates={}, n_epoches={}".format(n_components, lr, n_epoches))
    #     plt.show()
    def plt_losses(self, n_components, lr, n_epoches, std_idx, train=True, holdout=True):
        assert len(self.train_losses) == len(self.holdout_losses)
        images_dir = "../Report/Images/"
        plt.figure()
        mask = np.zeros(n_epoches)
        std_idx = [i - 1 for i in std_idx]
        mask[std_idx] = 1
        if train:
            train_yerror = np.std(self.train_losses, axis=0) * mask
            plt.errorbar(range(n_epoches), np.mean(self.train_losses, axis=0), yerr=train_yerror, label='train')
        if holdout:
            holdout_yerror = np.std(self.train_losses, axis=0) * mask
            plt.errorbar(range(n_epoches), np.mean(self.holdout_losses, axis=0), yerr=holdout_yerror,
                         label='holdout')
        plt.title("n_components={}, learning_rates={}, n_epoches={}".format(n_components, lr, n_epoches))
        plt.xlabel("Epoches")
        plt.ylabel("Cross Entropy Loss")
        plt.legend()
        #         plt.show()
        plt.savefig(images_dir + "{} losses, n_components={}, learning_rates={}, n_epoches={}.png".format(
            'softmax', n_components, lr, n_epoches),
                    bbox_inches='tight')

    def show_accuracies(self):
        print("The accuracy is {}".format(np.mean(self.test_accuracies)))

    def plt_confusion(self):
        plt.matshow(self.test_confusion)

    def print_confusion(self, classifier_type, n_components, lr, n_epoches, n_labels): #, confusion_matrix):
        print(
            "{} confusion, n_components={}, learning_rates={}, n_epoches={}".format(classifier_type, n_components,
                                                                                    lr, n_epoches))
        emotion_dict = {"h": "happy", "ht": "happy with teeth", "m": "maudlin",
            "s": "surprise", "f": "fear", "a": "anger", "d": "disgust", "n": "neutral"}
        # expressions = ["disgust", "anger", "surprise","happy", "fear", "maudlin"]
        expressions = [emotion_dict[label] for label in n_labels]
        table_str = ""
        for i, line in enumerate(self.test_confusion):
            table_str += "\\hline\n" + expressions[i] + " & " + " & ".join([str(num) for num in line]) + "\\\\\n"
        return table_str