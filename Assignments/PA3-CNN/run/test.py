from models.intensive_cnn import *
from run.baseline_cnn import BasicCNN
from utils.Evaluation import *


def main(model_name, model_path):

    conf = {}
    conf['z_score'] = True

    # Setup: initialize the hyperparameters/variables
    num_epochs = 5           # Number of full passes through the dataset
    batch_size = 128           # Number of samples in each minibatch
    learning_rate = 1e-5
    seed = np.random.seed(1) # Seed the random number generator for reproducibility
    p_val = 0.1              # Percent of the overall dataset to reserve for validation
    p_test = 0.2             # Percent of the overall dataset to reserve for testing
    val_every_n = 100        #

    early_stop_counter = 0
    early_stop_max = 7
    is_converged = False

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

    train_loader, val_loader, test_loader, _ = create_balanced_split_loaders(batch_size, seed, transform=transform,
                                                                          p_val=p_val, p_test=p_test,
                                                                          shuffle=True, show_sample=False,
                                                                          extras=extras, z_score=conf['z_score'])


    if model_name == 'intensive':
        model = IntensiveCNN()
        model = model.to(computing_device)
        model.load_state_dict(torch.load(model_path)['model_state_dict'])
    elif model_name == 'baseline':
        model = BasicCNN()
        model = model.to(computing_device)
        model.load_state_dict(torch.load(model_path))
    model.eval()

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
    eval.evaluate()



if __name__ == "__main__":
    model_name = 'baseline'
    model_path = 'D:\model-online/baseline/baseline-model.pt'
    main(model_name, model_path)