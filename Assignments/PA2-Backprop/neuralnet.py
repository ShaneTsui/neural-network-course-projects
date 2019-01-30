import numpy as np
import pickle

config = {}
config['layer_specs'] = [784, 50, 50, 10]  # The length of list denotes number of hidden layers; each element denotes number of neurons in that layer; first element is the size of input layer, last element is the size of output layer.
config['activation'] = 'sigmoid' # Takes values 'sigmoid', 'tanh' or 'ReLU'; denotes activation function for hidden layers
config['batch_size'] = 1000  # Number of training samples per batch to be passed to network
config['epochs'] = 200  # Number of epochs to train the model
config['early_stop'] = True  # Implement early stopping or not
config['early_stop_epoch'] = 5  # Number of epochs for which validation loss increases to be counted as overfitting
config['L2_penalty'] = 0.0001  # Regularization constant
config['momentum'] = True  # Denotes if momentum is to be applied or not
config['momentum_gamma'] = 0.9  # Denotes the constant 'gamma' in momentum expression
config['learning_rate'] = 0.02 # Learning rate of gradient descent algorithm

# Be aware of overflow
def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def load_data(fname):
    """
    Write code to read the data and return it as 2 numpy arrays.
    Make sure to convert labels to one hot encoded format.
    """

    # Load data
    with open(fname, 'rb') as f:
        dataset = pickle.load(f)

    images, labels = [], []

    for data in dataset:
        images.append(data[:-1])
        labels.append(int(data[-1]))

    # Convert to one-hot encode
    labels_one_hot = np.zeros(shape=(len(images), 10))

    for idx, label in enumerate(labels):
        labels_one_hot[idx][label] = 1

    return np.array(images), labels_one_hot

class Activation:
    def __init__(self, activation_type = "sigmoid"):
        self.activation_type = activation_type
        self.x = None # Save the input 'x' for sigmoid or tanh or ReLU to this variable since it will be used later for computing gradients.
        self.output = None # Save the input 'y' for sigmoid or tanh to this variable since it will be used later for computing gradients.

    def forward_pass(self, a):
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

    def backward_pass(self, delta):
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()

        return grad * delta

    def sigmoid(self, x):
        self.x = x
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def tanh(self, x):
        self.x = x
        self.output = np.tanh(x)
        return self.output

    def ReLU(self, x):
        self.x = x
        self.output = x * (x > 0) # np.maximum(0, x)
        return self.output

    def grad_sigmoid(self):
        return self.output * (1 - self.output)

    def grad_tanh(self):
        return 1 - self.output * self.output

    def grad_ReLU(self):
        return 1. * (self.x > 0) # self.x > 0


class Layer():
    def __init__(self, in_units, out_units):
        np.random.seed(42)
        self.w = np.random.randn(in_units, out_units)  # Weight matrix
        self.b = np.zeros((1, out_units)).astype(np.float32)  # Bias
        self.x = None  # Save the input to forward_pass in this
        self.a = None  # Save the output of forward pass in this (without activation)

        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this
        self.delta_w_old = 0
        self.delta_b_old = 0

        self.w_snapshot = None # Weight matrix
        self.b_snapshot = None # Bias

    def forward_pass(self, x):
        """
        Takes input data of shape [batch, input_units], returns output data [batch, output_units].
        Do not apply activation function here.
        """
        self.x = x
        self.a = np.dot(x, self.w) + self.b
        return self.a

    def backward_pass(self, delta, l2_penalty=0):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        """
        N = self.x.shape[0]

        self.d_x = delta.dot(self.w.T)
        self.d_w = self.x.T.dot(delta) / N + l2_penalty * self.w
        self.d_b = delta.sum(axis=0) / N + l2_penalty * self.b

        return self.d_x

    def update_parameters(self, learning_rate, use_momentum=False, momentum_gamma=None):
        if use_momentum:
            delta_w = - learning_rate * self.d_w + momentum_gamma * self.delta_w_old
            delta_b = - learning_rate * self.d_b + momentum_gamma * self.delta_b_old

            self.w += delta_w
            self.b += delta_b

            self.delta_w_old = delta_w
            self.delta_b_old = delta_b
        else:
            self.w -= learning_rate * self.d_w
            self.b -= learning_rate * self.d_b

    def take_snapshot(self):
        self.w_snapshot = self.w  # Weight matrix
        self.b_snapshot = self.b  # Bias

    def load_snapshot(self):
        self.w = self.w_snapshot  # Weight matrix
        self.b = self.b_snapshot  # Bias

class Neuralnetwork():
    def __init__(self, config):
        self.layers = []
        self.x = None  # Save the input to forward_pass in this
        self.y = None  # Save the output vector of model in this
        self.targets = None  # Save the targets in forward_pass in this variable
        self.l2_penalty = None
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append( Layer(config['layer_specs'][i], config['layer_specs'][i+1]) )
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))


    def loss_func(self, logits, targets):
        '''
        find cross entropy loss between logits and targets
        '''
        N = targets.shape[0]
        return - np.sum(np.multiply(targets, np.log(logits))) / N


    def forward_pass(self, x, targets=None, l2_penalty=0):
        """
        Write the code for forward pass through all layers of the model and return loss and predictions.
        If targets == None, loss should be None. If not, then return the loss computed.
        """
        self.x = x
        self.targets = targets

        # Forwand pass
        input = x
        for layer in self.layers:
            input = layer.forward_pass(input)

        # Softmax
        self.y = softmax(input)

        # Compute cross entropy loss
        loss = self.loss_func(self.y, targets)

        if l2_penalty:
            for layer in self.layers:
                if isinstance(layer, Layer):
                    loss += (np.sum(layer.w ** 2) + np.sum(layer.b ** 2)) * l2_penalty / 2
        return loss

    def backward_pass(self, l2_penalty=0):
        '''
        implement the backward pass for the whole network.
        hint - use previously built functions.
        '''
        delta = self.y - self.targets
        for layer in self.layers[::-1]:
            if isinstance(layer, Layer):
                delta = layer.backward_pass(delta, l2_penalty)
            else:
                delta = layer.backward_pass(delta)

    def update_parameters(self, learning_rate, use_momentum=False, momentum_gamma=None):
        for layer in self.layers[::-1]:
            if isinstance(layer, Layer):
                layer.update_parameters(learning_rate, use_momentum, momentum_gamma)

    def take_snapshot(self):
        for layer in self.layers:
            if isinstance(layer, Layer):
                layer.take_snapshot()

    def load_snapshot(self):
        for layer in self.layers:
            if isinstance(layer, Layer):
                layer.load_snapshot()

    def predict(self, x, targets):
        input = x
        for layer in self.layers:
            input = layer.forward_pass(input)
        predictions = np.argmax(softmax(input), axis=1)
        targets = np.argmax(targets, axis=1)
        return np.mean(predictions == targets)


def batch_generator(inputs, targets, batch_size, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

def trainer(model, X_train, y_train, X_valid, y_valid, config):
    """
    Write the code to train the network. Use values from config to set parameters
    such as L2 penalty, number of epochs, momentum, etc.
    """

    epoches = config['epochs']  # Number of epochs to train the model
    batch_size = config['batch_size']
    use_momentum = config['momentum']  # Denotes if momentum is to be applied or not
    momentum_gamma = config['momentum_gamma']  # Denotes the constant 'gamma' in momentum expression
    learning_rate = config['learning_rate']  # Learning rate of gradient descent algorithm
    early_stop = config['early_stop']  # Implement early stopping or not
    early_stop_epoch = config['early_stop_epoch']  # Number of epochs for which validation loss increases to be counted as overfitting
    l2_penalty = config['L2_penalty']  # Regularization constant

    loss_val_min, no_increase_epoches = float('inf'), 0

    for epoch in range(epoches):
        for x, targets in batch_generator(X_train, y_train, batch_size=batch_size):
            loss_train = model.forward_pass(x, targets=targets, l2_penalty=l2_penalty)
            model.backward_pass(l2_penalty=l2_penalty)
            model.update_parameters(learning_rate=learning_rate, use_momentum=use_momentum, momentum_gamma=momentum_gamma)

        # Check accuracy
        loss_val = model.forward_pass(X_valid, targets=y_valid)
        train_acc = model.predict(x, targets=targets)
        loss_acc = model.predict(X_valid, targets=y_valid)

        print("Epoch {}, Train loss = {}, Train acc = {}, Val Loss = {}, Val acc = {}".format(epoch +1, loss_train, train_acc, loss_val,
                                                                                    loss_acc))
        # Check validation loss
        if loss_val < loss_val_min:
            model.take_snapshot()
            loss_val_min = loss_val
            no_increase_epoches = 0
        else:
            no_increase_epoches += 1

        if early_stop:
            if no_increase_epoches > early_stop_epoch:
                break

    return model

def test(model, X_test, y_test, config):
    """
    Write code to run the model on the data passed as input and return accuracy.
    """
    return model.predict(X_test, y_test)


if __name__ == "__main__":
    train_data_fname = './data/MNIST_train.pkl'
    valid_data_fname = './data/MNIST_valid.pkl'
    test_data_fname = './data/MNIST_test.pkl'

    ### Train the network ###
    model = Neuralnetwork(config)
    X_train, y_train = load_data(train_data_fname)
    X_valid, y_valid = load_data(valid_data_fname)
    X_test, y_test = load_data(test_data_fname)
    trainer(model, X_train, y_train, X_valid, y_valid, config)
    model.load_snapshot()
    test_acc = test(model, X_test, y_test, config)
    print(test_acc)

