import csv

def plot_loss(train_loss, val_loss, filename='loss'):
    # assert len(train_loss) == len(val_loss)
    n = len(train_loss)
    plt.plot(range(n), train_loss, label='train')
    plt.plot(range(n), val_loss, label='validation')
    plt.ylabel('loss')
    plt.xlabel('iteration')
    plt.legend()
    plt.title('loss curve')
    plt.savefig(filename)


fname = 'paper_loss_log_online.txt'

minibatch = 0
train_x, train_loss = [], []
val_x, val_loss = [], []
with open(fname) as f:
    for line in f:
        words = line.strip('\n').split(' ')
        if words[0] == 'Epoch':
            train_x.append(minibatch)
            train_loss.append(float(words[-1]))
            minibatch += 50
        elif words[0] == 'mini':
            val_x.append(minibatch)
            val_loss.append(float(words[-1]))

with open('train.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for tx, tl in zip(train_x, train_loss):
        writer.writerow([tx, tl])

with open('val.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for vx, vl in zip(val_x, val_loss):
        writer.writerow([vx, vl])
