import torch
import numpy as np
import matplotlib.pyplot as plt

from Evaluation import *

def kernel_visualize(model, idx, filename='kernel'):
    # visualize teh kernel of the first convolutional layer
    kernels = model.conv1.weight.data.numpy()
    plt.imshow(kernels[idx][0])
    plt.title('kernel visualization')
    plt.colorbar()
    plt.savefig(filename)


def plot_confusion(eval, filename='confusion'):
    matrix = eval.confusion_matrix()
    plt.matshow(matrix)
    plt.title('confusion matrix')
    plt.colorbar()
    plt.savefig(filename)
