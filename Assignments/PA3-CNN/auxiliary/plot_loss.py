import numpy as np
import matplotlib.pyplot as plt

def plot_loss(train_x, train_y, val_x, val_y, filename='loss'):
    plt.plot(train_x, train_y, label='train')
    plt.plot(val_x, val_y, label='validation')
    plt.ylabel('loss')
    plt.xlabel('iteration')
    plt.legend()
    plt.title('loss curve')
    plt.savefig(filename)


def reader(fname):
    x, y = [], []
    with open(fname) as f:
        for line in f:
            content = line.strip('\n').split(',')
            x.append(int(content[0]))
            y.append(float(content[1]))
        return np.array(x) / 50, np.array(y)

train_x, train_y = reader('./train.csv')
val_x, val_y = reader('./val.csv')

plot_loss(train_x, train_y, val_x, val_y, filename='loss')