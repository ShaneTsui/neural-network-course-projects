import numpy as np
import pickle

# import neuralnet
from neuralnet import Neuralnetwork, Activation, Layer, softmax, load_data

config = {}
config['layer_specs'] = [784, 50, 50, 10]  # The length of list denotes number of hidden layers; each element denotes number of neurons in that layer; first element is the size of input layer, last element is the size of output layer.
config['activation'] = 'ReLU' # Takes values 'sigmoid', 'tanh' or 'ReLU'; denotes activation function for hidden layers
config['batch_size'] = 1000  # Number of training samples per batch to be passed to network
config['epochs'] = 300  # Number of epochs to train the model
config['early_stop'] = True  # Implement early stopping or not
config['early_stop_epoch'] = 5  # Number of epochs for which validation loss increases to be counted as overfitting
config['L2_penalty'] = 0.0001  # Regularization constant
config['momentum'] = True  # Denotes if momentum is to be applied or not
config['momentum_gamma'] = 0.9  # Denotes the constant 'gamma' in momentum expression
config['learning_rate'] = 0.15 # Learning rate of gradient descent algorithm

def check_d_b(e, output_idx):
    layer.b[0][output_idx] += e
    loss_up = model.forward_pass(x, targets=targets)
    layer.b[0][output_idx] -= 2*e
    loss_down = model.forward_pass(x, targets=targets)
    num_d_b = (loss_up - loss_down) / (2 * e)
    layer.b[0][output_idx] += e
    return num_d_b

def check_d_w(e, input_idx, output_idx):
    layer.w[input_idx][output_idx] += e
    loss_up = model.forward_pass(x, targets=targets)
    layer.w[input_idx][output_idx] -= 2*e
    loss_down = model.forward_pass(x, targets=targets)
    num_d_w = (loss_up - loss_down) / (2 * e)
    layer.w[input_idx][output_idx] += e
    return num_d_w

# load data
train_data_fname = '../Assignments 2/PA2-Backprop/data/MNIST_train.pkl'
x, targets = load_data(train_data_fname)

model = Neuralnetwork(config)

loss_train = model.forward_pass(x, targets=targets)
model.backward_pass()

e = 0.01

d_w_lst, d_b_lst = [], []
for layer in model.layers:
    if isinstance(layer, Activation):
        continue
    d_b = check_d_b(e, 0)
    d_w_1 = check_d_w(e, 0, 0)
    d_w_2 = check_d_w(e, 0, 1)
    d_b_lst.append(d_b)
    d_w_lst.append([d_w_1, d_w_2])
    print('b gradient: ', layer.d_b[0], 'numerical estimation: ', d_b)
    print('w1 gradient: ', layer.d_w[0][0], 'numerical estimation: ', d_w_1)
    print('w2 gradient: ', layer.d_w[0][1], 'numerical estimation: ', d_w_2)



    