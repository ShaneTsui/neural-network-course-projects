from models.intensive_cnn import *
from run.baseline_cnn import BasicCNN
from utils.Evaluation import *
from PIL import Image
import math


def main(model_name, model_path):
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

    if model_name == 'intensive':
        model = IntensiveCNN()
    elif model_name == 'baseline':
        model = BasicCNN()
    model = model.to(computing_device)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()
    print(model)
    # kernels = model.features_d.cpu().conv0.weight.data.numpy()
    kernels = model.features.cpu().intensiveblock4.intensive15.branch5x5_2.conv.weight.data.numpy()
    # kernels = model.features.cpu().intensiveblock2.intensive1.branch5x5_2.conv.weight.data.numpy()
    print(kernels.shape)
    fig, ax = plt.subplots(nrows=kernels.shape[-1] * kernels.shape[-2] // 5,ncols=5)
    nrows = math.ceil(kernels.shape[0] / 5)
    ncols = 5
    for i in range(kernels.shape[0]):
        im_arr = kernels[i][0]
        im_arr = (im_arr - np.min(im_arr)) / (np.max(im_arr) - np.min(im_arr))
        img = Image.fromarray(np.uint8(im_arr * 255) , 'L')
        plt.subplot(nrows,ncols,i + 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    for i in range(i, ncols*nrows):
        plt.subplot(nrows, ncols, i + 1)
        plt.axis('off')
    plt.show()
        # img.show()

    # plt.imshow(kernels[0][0])
    # plt.show()
    # plt.title('kernel visualization')
    # plt.colorbar()
    # plt.savefig('visualize')
    # print(weights)


if __name__ == "__main__":
    model_name = 'intensive'
    model_path = 'D:\model-online\epoch_4-batch_0-loss_1.0026615858078003-20190219-074201.pt'
    main(model_name, model_path)