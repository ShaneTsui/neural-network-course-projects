from models.intensive_cnn import *
from baseline_cnn import BasicCNN
from utils.Evaluation import *
from models.resnet_variant import resnet_n2


def main(model_list):

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

    # Load model
    models = {}
    for model_name, model_path in model_list:
        if model_name == 'intensive':
            models[model_name] = IntensiveCNN().to(computing_device)
            models[model_name].load_state_dict(torch.load(model_path)['model_state_dict'])
        elif model_name == 'baseline':
            models[model_name] = BasicCNN().to(computing_device)
            models[model_name].load_state_dict(torch.load(model_path))
        elif model_name == 'resnet':
            models[model_name] = resnet_n2(pretrained=False, num_classes=14).to(computing_device)
            models[model_name].load_state_dict(torch.load(model_path))
        models[model_name].eval()

    model_names = list(models.keys())
    num_models = len(model_name)
    labels_all = []
    predictions_all = []
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(computing_device), labels.to(computing_device)
            labels_all.append(labels)

            predictions = torch.zeros_like(labels).to(computing_device)
            for model_name in model_names:
                if model_name == 'resnet':
                    output = models[model_name](images.repeat(1, 3, 1, 1))
                else:
                    output = models[model_name](images)
                predictions += output
            pred = (predictions / num_models) > 0.5
            print(pred)
            predictions_all.append(pred)

    labels = torch.cat(labels_all, 0)
    predctions = torch.cat(predictions_all, 0)

    print(labels, predictions)

    eval = Evaluation(predctions.float(), labels)
    eval.evaluate()



if __name__ == "__main__":
    baseline_path = 'D:\model-online/baseline/baseline-model.pt'
    intensive_model_path = 'D:\model-online\epoch_4-batch_0-loss_1.0026615858078003-20190219-074201.pt'
    resnet_model_path = 'D:\model-online/resnet\epoch_1-batch_0-loss_1.0875942707061768-20190218-185232.pt'
    models = [('intensive', intensive_model_path), ('resnet', resnet_model_path), ('baseline', baseline_path)]
    main(models)

    # img = torch.FloatTensor([[[[1,1,1],[2,2,2],[3,3,3]]], [[[1,1,1],[2,2,2],[3,3,3]]]])
    # concat = img.repeat(1,3,1,1)
    # print(concat)
    #
    # img = torch.FloatTensor([[[1,1,1],[2,2,2],[3,3,3]]])
    # con = torch.cat([img, img, img], 0)
    # print(con)