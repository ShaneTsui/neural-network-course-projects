# Part 1 How to run the code
You should download the data by following this link: https://github.com/cse253-neural-networks/PA2-Backprop.
And both the data folder and validate_data.pkl should be put at the root directory.

The main part of the code shall be contained in neutralnet.py.

In order to tailer the code to meet the requirement of the instructions, you only need to change the configure variable at the top of the neuralnet.py.
To generate figure used in the paper, you can use the fig() function in neuralnet.py.

To generate the answer for PA II (b), you can simply run p2.py file to reproduce the required result.

# Part 2 The design of the class
## Dataset
Unzip ```data.zip``` to get the pickle files for train, validation and test splits of MNIST dataset. The data is in the form of ```n * 785``` NumPy array (in which the first 784 columns contain the flattend 28 * 28 MNIST image and the last column gives the class of image from 0 to 9. All of the splits have been shuffled so you may skip the shuffling step. You need to implment the function ```load_data``` to return 2 arrays X, Y given a pickle file. X should be the input features and Y should be the one-hot encoded labels of each input image i.e ```shape(X) = n,784``` and ```shape(Y) = n,10```


## Activation Functions
There are 3 activation functions (sigmoid, ReLU and tanh) and their gradients which you will implement and experiment with. The gradient of the output of an activation unit with respect to the input will be multiplied by the upstream gradient during the backward pass to be passed on to the previous layer. 


## Layers
Similar to the activation functions, you will be implementing the linear layers of the neural network. The forward pass of a layer can be implemented as matrix multiplication of the weights with inputs and addition of biases. In the backward pass, given the gradient of the loss with respect to the output of the layer (delta), we need to compute the gradient of the loss with respect to the inputs of the layer and with respect to the weights and biases. The gradient with respect to the inputs will be passed on to the previous layers during backpropagation.


## Neural Network
Having implemented the lower level abstractions of layers and activation functions, the next step is to implment the forward and backward pass of the neural network by iteratively going through the layers and activations of the network. Remember that we will be caching the inputs to each layer and activation function during the forward pass (in self.x) and using it during the backward pass to compute the gradients. 

## Training
You will implement the training procedure in the trainer function. The network will be trained for ```config['epochs']``` epochs over the dataset in mini-batches of ```size config['batch_size']```. During each iteration (e.g. each mini-batch) you will call the forward and backward pass of the neural network and use the gradients ```layer.d_w, layer.d_b``` to update the weights and biases ```layer.w, layer.b``` of each layer in the network according to the update rule. Note that activation layers don't have any associated parameters.
