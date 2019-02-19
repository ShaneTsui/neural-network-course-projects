import numpy as np
import torch

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import matplotlib.pyplot as plt


class Evaluation:

    # torch : k-hot encoding
    def __init__(self, predicts, targets):
        self.confusion = None
        self.predicts = predicts
        self.targets = targets

        #   1     1 / 1 (True Positive)
        #   inf   1 / 0 (False Positive)
        #   nan   0 / 0 (True Negative)
        #   0     0 / 1 (False Negative)
        confusion_vector = predicts / targets

        self.TP = torch.sum(confusion_vector == 1, dim=0).double()
        self.FP = torch.sum(confusion_vector == float('inf'), dim=0).double()
        self.TN = torch.sum(torch.isnan(confusion_vector), dim=0).double()
        self.FN= torch.sum(confusion_vector == 0, dim=0).double()

        self.acc = torch.sum((self.predicts == self.targets).double(), dim=0) / self.predicts.shape[0]
        self.pre = self.TP / (self.TP + self.FP)
        self.rec = self.TP / (self.TP + self.FN)
        self.bcr = (self.pre + self. rec) / 2

        self.acc[torch.isnan(self.acc)] = 0
        self.pre[torch.isnan(self.pre)] = 0
        self.rec[torch.isnan(self.rec)] = 0
        self.bcr[torch.isnan(self.bcr)] = 0

        print(self.TP, self.FP, self.TN, self.FN)

    def accuracy(self):
        return self.acc

    def precision(self):
        return self.pre

    def recall(self):
        return self.rec

    def BCR(self):
        return self.bcr

    def avg_accuracy(self):
        return torch.mean(self.acc)

    def avg_precision(self):
        return torch.mean(self.pre)

    def avg_recall(self):
        return torch.mean(self.rec)

    def avg_BCR(self):
        return torch.mean(self.bcr)

    '''
    def confusion_matrix(self):
        
            pred | tar |   Action
              0  |  0  |   + (no disease, no disease)
              0  |  1  |   + (current, no disease presented)
              1  |  1  |   + (current, current)
              1  |  0  |   + all (current, other ground truth)
        
        num_classes = self.predicts.shape[1] + 1
        matrix = np.zeros(shape=(num_classes, num_classes))
        for predicts, targets in zip(self.predicts, self.targets):
            for cls, (pred, target) in enumerate(zip(predicts, targets)):
                if pred:
                    if target:
                        matrix[cls][cls] += 1
                    else:
                        n_wrong = sum(targets)
                        for tgt_cls, tar in enumerate(targets):
                            if tar:
                                matrix[cls][tgt_cls] += 1/n_wrong
                else:
                    if target:
                        matrix[-1][cls] += 1
                    else:
                        matrix[-1][-1] += 1
        self.confusion = matrix / self.predicts.shape[0]
        return self.confusion
        '''

    def confusion_matrix(self):
        d = self.targets.shape[1]
        conf = np.zeros(shape=(d+1, d+1))

        for (y_true, y_pred) in zip(self.targets, self.predicts):
            y_true = y_true.cpu().numpy()
            y_pred = y_pred.cpu().numpy()

            indices_tar = set(y_true.nonzero()[0])
            indices_pred = set(y_pred.nonzero()[0])
            intersection = indices_tar & indices_pred

            # target no disease, predcition no disease
            if len(indices_tar) == 0 and len(indices_pred) == 0:
                conf[d][d] += 1
            # target no disease, predcition has
            elif len(indices_tar) == 0 and len(indices_pred) > 0:
                for ind in indices_pred:
                    conf[ind][d] += 1

            # target has, prediction no disease
            elif len(indices_tar) and len(indices_pred) == 0:
                for ind in indices_pred:
                    conf[d][ind] += 1

            # target has, prediction has
            else:
                if len(intersection) == 0:
                    for i in indices_pred:
                        for j in indices_tar:
                            conf[i][j] += 1
                else:
                    for k in intersection:
                        conf[k][k] += 1
                    tar2 = indices_tar - intersection
                    pred2 = indices_pred - intersection
                    if len(tar2) == 0 and len(pred2) == 0:
                        continue
                    elif len(tar2) == 0 and len(pred2) > 0:
                        for ind in pred2:
                            conf[ind][d] += 1
                    elif len(tar2) > 0 and len(pred2) == 0:
                        for ind in tar2:
                            conf[d][ind] += 1
                    else:
                        for i in pred2:
                            for j in tar2:
                                conf[i][j] += 1

        self.confusion = conf / sum(conf)
        return conf

    def heatmap(self, data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
        if not ax:
            ax = plt.gca()

        # Plot the heatmap
        im = ax.imshow(data, **kwargs)

        # Create colorbar
        # cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)

        ax.tick_params(top=True, bottom=False,
                       labeltop=True, labelbottom=False)

        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                 rotation_mode="anchor")

        for edge, spine in ax.spines.items():
            spine.set_visible(False)

        ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
        ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        return im

    def annotate_heatmap(self, im, data=None, valfmt="{x:.2f}",
                         textcolors=["black", "white"],
                         threshold=None, **textkw):
        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()

        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max()) / 2.

        kw = dict(horizontalalignment="center",
                  verticalalignment="center")
        kw.update(textkw)

        if isinstance(valfmt, str):
            valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)

        return texts

    def plot_confusion_matrix(self, confusion=None):
        if confusion is None:
            confusion = self.confusion

        if confusion is None:
            confusion = self.confusion_matrix()

        predict = ["Atelectasis", "Cardiomegaly", "Effusion",
                   "Infiltration", "Mass", "Nodule", "Pneumonia",
                   "Pneumothorax", "Consolidation", "Edema",
                   "Emphysema", "Fibrosis",
                   "Pleural_Thickening", "Hernia", "No Findings"]
        target = ["Atelectasis", "Cardiomegaly", "Effusion",
                  "Infiltration", "Mass", "Nodule", "Pneumonia",
                  "Pneumothorax", "Consolidation", "Edema",
                  "Emphysema", "Fibrosis",
                  "Pleural_Thickening", "Hernia", "No Findings"]

        fig, ax = plt.subplots(figsize=(15, 15))

        im = self.heatmap(confusion, predict, target, ax=ax, cmap="YlGn", cbarlabel="confusion matrix")
        texts = self.annotate_heatmap(im, valfmt="{x:.4f} t")

        fig.tight_layout()
        plt.show()

    def evaluate(self):
        print('accuracy:', self.accuracy())
        print('precision:', self.precision())
        print('recall:', self.recall())
        print('BCR:', self.BCR())

        print('avg_accuracy:', self.avg_accuracy())

        print('average_precision:', self.avg_precision())
        print('average_recall:', self.avg_recall())
        print('average_BCR:', self.avg_BCR())
        print('confusion_matrix', self.confusion_matrix())

        print("plot the confusion matrix: ")
        self.plot_confusion_matrix()


if __name__=="__main__":

    prediction = torch.tensor([
        [1, 0, 0., 1],
        [1, 1., 0., 1],
        [0, 1., 0., 1]
    ])

    target = torch.tensor([
        [1, 1, 0., 1],
        [1, 0, 1., 1],
        [1, 1., 0., 1]
    ])

    eva = Evaluation(prediction, target)
    eva.evaluate()