from models.intensive_cnn import *
import torch.optim as optim
import time
import os
from utils.Evaluation import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.loss import weighted_loss, w_cel_loss


def main(loss_function):
    conf = {}
    conf['z_score'] = True

    print(os.getcwd())

    cur_time = '20190218-124848'
    model_path = 'D:/model-online/'
    result_path = 'D:/results/intensive/20190218-124848/'
    model = 'epoch_1-batch_2999-loss_1.052208423614502-20190218-214334.pt'
    PATH = model_path + model

    # Setup: initialize the hyperparameters/variables
    num_epochs = 5  # Number of full passes through the dataset
    batch_size = 16  # Number of samples in each minibatch
    learning_rate = 1e-5
    seed = np.random.seed(1)  # Seed the random number generator for reproducibility
    p_val = 0.1  # Percent of the overall dataset to reserve for validation
    p_test = 0.2  # Percent of the overall dataset to reserve for testing
    val_every_n = 2000  #

    early_stop_counter = 0
    early_stop_max = 10
    is_converged = False

    #     # Set up folder for model saving
    #     cur_time = time.strftime("%Y%m%d-%H%M%S")
    #     model_path = PATH #'D:/models/intensive/{}/'.format(cur_time)
    #     model_pathlib = pathlib.Path(model_path)
    #     if not model_pathlib.exists():
    #         pathlib.Path(model_pathlib).mkdir(parents=True, exist_ok=True)

    #     # Set up folder for model saving
    #     result_path = 'D:/results/intensive/{}/'.format(cur_time)
    #     result_pathlib = pathlib.Path(result_path)
    #     if not result_pathlib.exists():
    #         pathlib.Path(result_pathlib).mkdir(parents=True, exist_ok=True)

    # TODO: Convert to Tensor - you can later add other transformations, such as Scaling here
    transform = transforms.Compose([transforms.Resize(512), transforms.ToTensor()])

    # Check if your system supports CUDA
    use_cuda = torch.cuda.is_available()

    # Setup GPU optimization if CUDA is supported
    if use_cuda:
        computing_device = torch.device("cuda")
        extras = {"num_workers": 0, "pin_memory": True}
        print("CUDA is supported")
    else:  # Otherwise, train on the CPU
        computing_device = torch.device("cpu")
        extras = False
        print("CUDA NOT supported")

    train_loader, val_loader, test_loader, _ = create_balanced_split_loaders(batch_size, seed, transform=transform,
                                                                             p_val=p_val, p_test=p_test,
                                                                             shuffle=True, show_sample=False,
                                                                             extras=extras, z_score=conf['z_score'])

    # Instantiate a BasicCNN to run on the GPU or CPU based on CUDA support

    model = IntensiveCNN()
    model = model.to(computing_device)
    print("Model on CUDA?", next(model.parameters()).is_cuda)

    # TODO: Define the loss criterion and instantiate the gradient descent optimizer
    if loss_function == 'multi':
        criterion = nn.MultiLabelSoftMarginLoss()  # TODO - loss criteria are defined in the torch.nn package
    elif loss_function == 'weighted':
        criterion = weighted_loss()  # TODO - loss criteria are defined in the torch.nn package
    elif loss_function == 'wcel':
        criterion = w_cel_loss()  # TODO - loss criteria are defined in the torch.nn package

    # TODO: Instantiate the gradient descent optimizer - use Adam optimizer with default parameters
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate)  # TODO - optimizers are defined in the torch.optim package
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_last = checkpoint['epoch']
    loss = checkpoint['loss']
    #     early_stop_counter = checkpoint['early_stop_counter']

    # train: Track the loss across training
    total_loss = []
    avg_train_minibatch_loss = []

    # val: Track the loss across validation
    #     val_loss_min = checkpoint['val_loss_min']
    val_loss_min = 1.215
    avg_val_minibatch_loss = []

    # Begin training procedure
    for epoch in range(epoch_last, num_epochs):

        N = 50
        N_minibatch_loss = 0.0

        # Get the next minibatch of images, labels for training
        for minibatch_count, (images, labels) in enumerate(train_loader):

            # Put the minibatch data in CUDA Tensors and run on the GPU if supported
            images, labels = images.to(computing_device), labels.to(computing_device)

            # Zero out the stored gradient (buffer) from the previous iteration
            model.train()
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

            # print("minibatch {}, train loss {}".format(minibatch_count, loss.item()))

            # TODO: Implement holdout-set-validation
            if minibatch_count % val_every_n == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    for val_batch_count, (val_image, val_labels) in enumerate(val_loader, 1):
                        val_image, val_labels = val_image.to(computing_device), val_labels.to(computing_device)
                        val_outputs = model(val_image)
                        loss = criterion(val_outputs, val_labels)
                        val_loss += loss

                    val_loss /= (val_batch_count + 1)
                    print("mini batch {}, val loss{}".format(minibatch_count, val_loss))
                    avg_val_minibatch_loss.append(val_loss.cpu().numpy())
                    scheduler.step(val_loss)

                    if val_loss < val_loss_min:
                        model_name = "epoch_{}-batch_{}-loss_{}-{}.pt".format(epoch, minibatch_count, val_loss,
                                                                              time.strftime("%Y%m%d-%H%M%S"))
                        print("Model saved to {}".format(model_name))
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': val_loss,
                            'val_loss_min': val_loss_min,
                            'early_stop_counter': early_stop_counter
                        }, os.path.join(model_path, model_name))
                        # torch.save(model.state_dict(), os.path.join(model_path, model_name))
                        val_loss_min = val_loss
                        early_stop_counter = 0

                    else:
                        early_stop_counter += 1

                    if early_stop_counter >= early_stop_max:
                        is_converged = True

            if minibatch_count % N == 0:
                # Print the loss averaged over the last N mini-batches
                N_minibatch_loss /= N
                print('Epoch %d, average minibatch %d loss: %.3f' %
                      (epoch + 1, minibatch_count, N_minibatch_loss))

                # Add the averaged loss over N minibatches and reset the counter
                avg_train_minibatch_loss.append(N_minibatch_loss.item())
                N_minibatch_loss = 0.0

            if is_converged:
                break

        print("Finished", epoch + 1, "epochs of training")

        if is_converged:
            break

    print("Training complete after", epoch, "epochs")

    print("Writing result...")
    np.savetxt('{}{}_train_batch_{}loss.csv'.format(result_path, cur_time, loss_function), np.array(total_loss),
               delimiter=',')
    np.savetxt('{}{}_train_avg_{}loss.csv'.format(result_path, cur_time, loss_function),
               np.array(avg_train_minibatch_loss), delimiter=',')
    np.savetxt('{}{}_val_avg_{}loss.csv'.format(result_path, cur_time, loss_function), np.array(avg_val_minibatch_loss),
               delimiter=',')

    print("Done")
    # Begin testing
    # labels_all = []
    # predictions_all = []
    # model.eval()
    # with torch.no_grad():
    #     for data in test_loader:
    #         images, labels = data
    #         images, labels = images.to(computing_device), labels.to(computing_device)
    #         labels_all.append(labels)
    #         output = model(images)
    #         predictions = output > 0.5
    #         predictions_all.append(predictions)
    #
    # labels = torch.cat(labels_all, 0)
    # predctions = torch.cat(predictions_all, 0)
    #
    # eval = Evaluation(predctions.float(), labels)
    # print(eval.accuracy())
    # print(eval.accuracy().mean())


if __name__ == "__main__":
    for loss_func in ['wcel']:
        main(loss_func)