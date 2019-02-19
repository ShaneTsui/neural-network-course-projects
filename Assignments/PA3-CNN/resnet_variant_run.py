from resnet_variant import *
from baseline_cnn import *
from resnet_variant import resnet_n2
import torch.nn as nn
import torch.optim as optim
import time
import pathlib
import torch
from evaluation import Evaluation
from xray_imbalanced_dataloader import create_balanced_split_loaders
import random
from balanced_loss import w_cel_loss


num_epochs = 15           # Number of full passes through the dataset
batch_size = 64          # Number of samples in each minibatch
learning_rate = 0.00001  
seed = np.random.seed(1) # Seed the random number generator for reproducibility
p_val = 0.1              # Percent of the overall dataset to reserve for validation
p_test = 0.2             # Percent of the overall dataset to reserve for testing

class channelCopy(object):
    def __call__(self, img):
        return torch.cat([img, img, img], 0)

#TODO: Convert to Tensor - you can later add other transformations, such as Scaling here
transform = transforms.Compose([transforms.Resize(512),transforms.ToTensor(),channelCopy()])


# Check if your system supports CUDA
use_cuda = torch.cuda.is_available()

# Setup GPU optimization if CUDA is supported
if use_cuda:
    computing_device = torch.device("cuda")
    extras = {"num_workers": 1, "pin_memory": True}
    print("CUDA is supported")
else: # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    extras = False
    print("CUDA NOT supported")

train_loader, val_loader, test_loader, label_weights = create_balanced_split_loaders(batch_size, seed, transform=transform,                                                                 p_val=p_val, p_test=p_test,
                                                                 shuffle=True, show_sample=False,
                                                                 extras=extras, z_score=True)
# Instantiate a BasicCNN to run on the GPU or CPU based on CUDA support
model = resnet_n2(pretrained=False, num_classes=14)
model = model.to(computing_device)
print("Model on CUDA?", next(model.parameters()).is_cuda)

criterion = w_cel_loss()
#TODO: Instantiate the gradient descent optimizer - use Adam optimizer with default parameters
optimizer = optim.Adam(model.parameters()) #TODO - optimizers are defined in the torch.optim package


# Set up folder for model saving
model_path = '{}/models/resnet_n2/paperLoss/{}/'.format(os.getcwd(), time.strftime("%Y%m%d-%H%M%S"))
model_pathlib = pathlib.Path(model_path)
if not model_pathlib.exists():
    pathlib.Path(model_pathlib).mkdir(parents=True, exist_ok=True)

#save loss in files for futher usage.
loss_path = '{}/losses/resnet_n2/paperLoss/{}/'.format(os.getcwd(), time.strftime("%Y%m%d-%H%M%S"))
loss_pathlib = pathlib.Path(loss_path)
if not loss_pathlib.exists():
    pathlib.Path(loss_pathlib).mkdir(parents=True, exist_ok=True)

print('data prepated, start to train.')
# Track the loss across training
total_loss = []
avg_minibatch_loss = []
loss_val_list = []
loss_val_min = float('inf')
N = 50
M = 100
# Begin training procedure
for epoch in range(num_epochs):

    
    N_minibatch_loss = 0.0
    

    # Get the next minibatch of images, labels for training
    for minibatch_count, (images, labels) in enumerate(train_loader, 0):
#         if minibatch_count == 100:
#             break
        # Put the minibatch data in CUDA Tensors and run on the GPU if supported
        images, labels = images.to(computing_device), labels.to(computing_device)

        # Zero out the stored gradient (buffer) from the previous iteration
        optimizer.zero_grad()

        # Perform the forward pass through the network and compute the loss
        outputs = model(images)
        loss = criterion(outputs, labels)
#         print('training',minibatch_count,loss)
        # Automagically compute the gradients and backpropagate the loss through the network
        loss.backward()
        
        # Update the weights
        optimizer.step()

        # Add this iteration's loss to the total_loss
        total_loss.append(loss.item())
        N_minibatch_loss += loss
        
        #TODO: Implement validation
        if minibatch_count % M == 0:
            if minibatch_count > 200:
                M = 500
            #switch to evaluate mode
            model.eval()
            with torch.no_grad():
                loss_val = 0
                for count_val, (images_val, labels_val) in enumerate(val_loader):
#                     if count_val ==10:
#                         break
                    images_val, labels_val = images_val.to(computing_device), labels_val.to(computing_device)
                    outputs_val = model(images_val)
                    loss_val += criterion(outputs_val, labels_val)
#                     print('val',count_val, (loss_val/(count_val+1)))
                loss_val /= (count_val+1)
                print('val',minibatch_count,loss_val)
                loss_val_list.append(loss_val.item())
                if loss_val < loss_val_min:
                    model_name = "epoch_{}-batch_{}-loss_{}-{}.pt".format(epoch, minibatch_count, loss_val, time.strftime("%Y%m%d-%H%M%S"))
                    torch.save(model.state_dict(), os.path.join(model_path, model_name))
                    loss_val_min = loss_val
                    
        if minibatch_count % N == 0:    
            
            # Print the loss averaged over the last N mini-batches    
            N_minibatch_loss /= N
            print('Epoch %d, average minibatch %d loss: %.6f' %
                (epoch + 1, minibatch_count, N_minibatch_loss))
            
            # Add the averaged loss over N minibatches and reset the counter
            avg_minibatch_loss.append(N_minibatch_loss.item())
            N_minibatch_loss = 0.0
            
        prefix = 'resnet_n2_balancedloss_'
        with open(os.path.join(loss_path, prefix+"training.txt"), "w") as f:
            for s in total_loss:
                f.write(str(s) +"\n")

        with open(os.path.join(loss_path, prefix+"training_ave.txt"), "w") as f:
            for s in avg_minibatch_loss:
                f.write(str(s) +"\n")

        with open(os.path.join(loss_path, prefix+"val.txt"), "w") as f:
            for s in loss_val_list:
                f.write(str(s) +"\n")
            
    print("Finished", epoch + 1, "epochs of training")
    model_name = "epoch_{}-{}.pt".format(epoch, time.strftime("%Y%m%d-%H%M%S"))
    torch.save(model.state_dict(), os.path.join(model_path, model_name))
print("Training complete after", epoch, "epochs")

