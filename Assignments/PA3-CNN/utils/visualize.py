from utils.Evaluation import *

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


def plot_loss(train_loss, val_loss, filename='loss'):
    assert len(train_loss) == len(val_loss)
    n = len(train_loss)
    plt.plot(range(n), train_loss, label='train')
    plt.plot(range(n), val_loss, label='validation')
    plt.ylabel('loss')
    plt.xlabel('iteration')
    plt.legend()
    plt.title('loss curve')
    plt.savefig(filename)
