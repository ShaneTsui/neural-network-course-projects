from baseline_cnn import *
from baseline_cnn import BasicCNN
import torch.nn as nn
import torch.optim as optim
import time
import pathlib
from loss import *


def main():
    conf = {}
    conf['z_score'] = False


    # Setup: initialize the hyperparameters/variables
    num_epochs = 1           # Number of full passes through the dataset
    batch_size = 16          # Number of samples in each minibatch
    learning_rate = 1e-5
    seed = np.random.seed(1) # Seed the random number generator for reproducibility
    p_val = 0.1              # Percent of the overall dataset to reserve for validation
    p_test = 0.2             # Percent of the overall dataset to reserve for testing
    val_every_n = 100
    batch_count = 0

    # Set up folder for model saving
    model_path = '{}/saved-models/baseline/{}/'.format(os.getcwd(), time.strftime("%Y%m%d-%H%M%S"))
    model_pathlib = pathlib.Path(model_path)
    if not model_pathlib.exists():
        pathlib.Path(model_pathlib).mkdir(parents=True, exist_ok=True)


    # TODO: Convert to Tensor - you can later add other transformations, such as Scaling here
    transform = transforms.Compose([transforms.Resize(512), transforms.ToTensor()])

    # Check if your system supports CUDA
    use_cuda = torch.cuda.is_available()

    # Setup GPU optimization if CUDA is supported
    if use_cuda:
        computing_device = torch.device("cuda")
        extras = {"num_workers": 0, "pin_memory": True}
        print("CUDA is supported")
    else: # Otherwise, train on the CPU
        computing_device = torch.device("cpu")
        extras = False
        print("CUDA NOT supported")

    # Setup the training, validation, and testing dataloaders
    train_loader, val_loader, test_loader, _ = create_balanced_split_loaders(batch_size, seed, transform=transform,
                                                                 p_val=p_val, p_test=p_test,
                                                                 shuffle=True, show_sample=False,
                                                                 extras=extras, z_score=conf['z_score'])

    # Instantiate a BasicCNN to run on the GPU or CPU based on CUDA support
    model = BasicCNN()
    model = model.to(computing_device)
    print("Model on CUDA?", next(model.parameters()).is_cuda)

    #TODO: Define the loss criterion and instantiate the gradient descent optimizer
    # criterion = nn.MultiLabelSoftMarginLoss() #TODO - loss criteria are defined in the torch.nn package
    # criterion = weighted_loss() #TODO - loss criteria are defined in the torch.nn package
    criterion = w_cel_loss() #TODO - loss criteria are defined in the torch.nn package

    #TODO: Instantiate the gradient descent optimizer - use Adam optimizer with default parameters
    optimizer = optim.Adam(model.parameters(), lr=0.001) #TODO - optimizers are defined in the torch.optim package

    # Track the loss across training
    total_loss = []
    avg_minibatch_loss = []
    val_loss_min = float('inf')

    # Begin training procedure
    for epoch in range(num_epochs):

        N = 50
        N_minibatch_loss = 0.0

        # Get the next minibatch of images, labels for training
        for minibatch_count, (images, labels) in enumerate(train_loader):

            # batch scheduler
            batch_count += 1
            if batch_count > 5000:
                val_every_n = 500
            elif batch_count > 10000:
                val_every_n = 1000
            elif batch_count > 15000:
                val_every_n = 1500
            elif batch_count > 20000:
                val_every_n = 2000

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
            print("Train loss = {}".format(loss.item()))

            # TODO: Implement holdout-set-validation
            if minibatch_count % val_every_n:
                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    for val_batch_count, (val_image, val_labels) in enumerate(val_loader):
                        val_image, val_labels = val_image.to(computing_device), val_labels.to(computing_device)
                        val_outputs = model(val_image)
                        val_loss += criterion(val_outputs, val_labels)
                    val_loss /= (val_batch_count + 1)
                    if val_loss < val_loss_min:
                        model_name = "epoch_{}-batch_{}-loss_{}-{}.pt".format(epoch, minibatch_count, val_loss, time.strftime("%Y%m%d-%H%M%S"))
                        torch.save(model.state_dict(), os.path.join(model_path, model_name))
                        val_loss_min = val_loss

            if minibatch_count % N == 0:
                # Print the loss averaged over the last N mini-batches
                N_minibatch_loss /= N
                print('Epoch %d, average minibatch %d loss: %.3f' %
                      (epoch + 1, minibatch_count, N_minibatch_loss))

                # Add the averaged loss over N minibatches and reset the counter
                avg_minibatch_loss.append(N_minibatch_loss)
                N_minibatch_loss = 0.0

        print("Finished", epoch + 1, "epochs of training")
    print("Training complete after", epoch, "epochs")


    # Begin testing
    correct = 0
    total = 0
    for data in test_loader:
        images, labels = data
        outputs = model(Variable(images.cuda()))
        predict = torch.sigmoid(outputs) > 0.5#.to(computing_device)
        l = labels.byte().to(computing_device)
        r = (l == predict)
        # Use Evaluation to get result

if __name__ == '__main__':
    main()