import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

def display_face(img):
    """ Display the input image and optionally save as a PNG.
    Args:
        img: The NumPy array or image to display
    Returns: None
    """
    img = img.reshape(380, 240)
    # Convert img to PIL Image object (if it's an ndarray)
    if type(img) == np.ndarray:
        img = Image.fromarray(img)

    # Display the image
    plt.imshow(np.asarray(img), cmap='gray')  # for jupyter notebook inline display
    plt.axis('off')


def display_faces(images, layout, labels):
    (n_row, n_col) = layout
    assert n_row * n_col == len(images) == len(labels)
    for i in range(len(images)):
        plt.subplot(n_row, n_col, i + 1)  # 1-6
        plt.title("{} face".format(labels[i]))
        display_face(images[i])


def display_subject(subject, layout=(2, 3)):
    (n_row, n_col) = layout
    for label in subject.label_image_dict.keys():
        print(subject.label_image_dict)
        plt.subplot(n_row, n_col, i + 1)  # 1-6
        plt.title("{} face".format(label))
        display_face(subject.label_image_dict[label])