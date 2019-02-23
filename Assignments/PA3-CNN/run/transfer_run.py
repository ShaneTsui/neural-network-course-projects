from data.xray_imbalanced_dataloader import *
from models.model_transfer import *
from utils.Evaluation import *

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import os
import time
import pathlib

from torchvision import transforms


def main():
    # import pretrained imagenet         
    imagenet = torchvision.models.resnet18(pretrained=True)

    # Set up folder for model saving
    model_path = '{}/models/new_trans/{}/'.format(os.getcwd(), time.strftime("%Y%m%d-%H%M%S"))
    model_pathlib = pathlib.Path(model_path)
    if not model_pathlib.exists():
        pathlib.Path(model_pathlib).mkdir(parents=True, exist_ok=True)


    # Check if your system supports CUDA
    use_cuda = torch.cuda.is_available()

    # Setup GPU optimization if CUDA is supported
    if use_cuda:
        computing_device = torch.device("cuda")
        extras = {"num_workers": 4, "pin_memory": True} # fix parameter: 5; finetuning: 
        print("CUDA is supported")
    else: # Otherwise, train on the CPU
        computing_device = torch.device("cpu")
        extras = False
        print("CUDA NOT supported")

    # Setup: initialize the hyperparameters/variables
    num_epochs = 1           # Number of full passes through the dataset
    batch_size = 32          # Number of samples in each minibatch
    seed = np.random.seed(1) # Seed the random number generator for reproducibility
    p_val = 0.1              # Percent of the overall dataset to reserve for validation
    p_test = 0.2             # Percent of the overall dataset to reserve for testing
    val_every_n = 50         #
    learning_rate = 0.0000001


    class channelCopy(object):
        
        def __call__(self, img):
            return torch.cat([img, img, img], 0)

    # TODO: Convert to Tensor - you can later add other transformations, such as Scaling here
    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), channelCopy()])

    # Setup the training, validation, and testing dataloaders
    train_loader, val_loader, test_loader, label_weights = create_balanced_split_loaders(batch_size, seed, transform=transform,
                                                                 p_val=p_val, p_test=p_test,
                                                                 shuffle=True, show_sample=False,
                                                                 extras=extras, z_score=True)
    # Instantiate a BasicCNN to run on the GPU or CPU based on CUDA support
    transfer = Transfer(14, finetuning=False)
    model = transfer(imagenet)
    model = model.to(computing_device)
    print("Model on CUDA?", next(model.parameters()).is_cuda)

    #TODO: Define the loss criterion and instantiate the gradient descent optimizer
    criterion = nn.MultiLabelSoftMarginLoss() #TODO - loss criteria are defined in the torch.nn package

    #TODO: Instantiate the gradient descent optimizer - use Adam optimizer with default parameters
    optimizer = optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=0.000002) #TODO - optimizers are defined in the torch.optim package

    # Track the loss across training
    total_loss = []
    avg_minibatch_loss = []
    avg_minibatch_val_loss = []
    val_loss_min = float('inf')
    
    
    # Begin training procedure
    for epoch in range(num_epochs):

        N = val_every_n
        N_minibatch_loss = 0.0

        # Get the next minibatch of images, labels for training
        for minibatch_count, (images, labels) in enumerate(train_loader):

            # Put the minibatch data in CUDA Tensors and run on the GPU if supported
            images, labels = images.to(computing_device), labels.to(computing_device)
            
            # Zero out the stored gradient (buffer) from the previous iteration
            optimizer.zero_grad()

            # Perform the forward pass through the network and compute the loss
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Automagically compute the gradients and backpropagate the loss through the network
            loss.backward()

            # Update the weights
            optimizer.step()

            # Add this iteration's loss to the total_loss
            total_loss.append(loss.item())
            N_minibatch_loss += loss
            
            print('training on {0} minibatch'.format(minibatch_count))
            
            # TODO: Implement holdout-set-validation
            if not minibatch_count % val_every_n:
                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    
                    
                    for val_batch_count, (val_image, val_labels) in enumerate(val_loader):   
                        
                        print('validating on {0} minibatch'.format(val_batch_count))
                        val_image, val_labels = val_image.to(computing_device), val_labels.to(computing_device)
                        val_outputs = model(val_image)
                        val_loss += criterion(val_outputs, val_labels)
 
                    val_loss /= val_batch_count
                    print('validation loss: {0}'.format(val_loss))
                    avg_minibatch_val_loss.append(val_loss)
                    model_name = "epoch_{}-batch_{}-loss_{}-{}.pt".format(epoch, minibatch_count, val_loss, time.strftime("%Y%m%d-%H%M%S"))
                    torch.save(model.state_dict(), os.path.join(model_path, model_name))
                    if val_loss < val_loss_min:
                        torch.save(model.state_dict(), os.path.join(model_path, 'best model'))
                        val_loss_min = val_loss
                    print('val: ', [l.item() for l in avg_minibatch_val_loss])

            if minibatch_count % N == 0:
                # Print the loss averaged over the last N mini-batches
                if minibatch_count > 0:
                    N_minibatch_loss /= N  
                print('Epoch %d, average minibatch %d loss: %.3f' %
                    (epoch + 1, minibatch_count, N_minibatch_loss))

                # Add the averaged loss over N minibatches and reset the counter
                avg_minibatch_loss.append(N_minibatch_loss)
                N_minibatch_loss = 0.0
                print('train: ', [l.item() for l in avg_minibatch_loss])

        print("Finished", epoch + 1, "epochs of training")
    print("Training complete after", epoch, "epochs")
    

    # Begin testing
    labels_all = []
    predictions_all = []
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(computing_device), labels.to(computing_device)
            labels_all.append(labels)
            output = model(images)
            predictions = output > 0.5
            predictions_all.append(predictions)

    labels = torch.cat(labels_all, 0)
    predctions = torch.cat(predictions_all, 0)

    eval = Evaluation(predctions.float(), labels)
    print('acc: ', eval.accuracy())
    print('acc: ', eval.accuracy().mean())
    print('pre: ', eval.precision())
    print('pre: ', eval.precision().mean())
    print('rec: ', eval.recall())
    print('rec: ', eval.recall().mean())
    
    

if __name__ == "__main__":
    main()
