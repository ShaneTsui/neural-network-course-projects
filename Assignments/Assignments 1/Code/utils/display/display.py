import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

emotion_dict = {"h": "happy", "ht": "happy with teeth", "m": "maudlin",
    "s": "surprise", "f": "fear", "a": "anger", "d": "disgust", "n": "neutral"}

def display_face(img, title):
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
    plt.title(title)
    plt.axis('off')

# Bad hardcode
def display_six_expressions(images, labels):
    selected_faces = set()
    different_faces = []
    different_labels = []
    for img, label in zip(images, labels):
        if label not in selected_faces:
            different_faces.append(img)
            different_labels.append(label)
            selected_faces.add(label)
    assert len(different_faces) == 6
    for i in range(6):
        plt.subplot(2, 3, i + 1)  # 1-6
        plt.title(emotion_dict[different_labels[i]])
        display_face(different_faces[i], emotion_dict[different_labels[i]])
    plt.show()

def display_faces(images, layout, labels):
    (n_row, n_col) = layout
    # assert n_row * n_col == len(images) == len(labels)
    for i in range(len(images)):
        plt.subplot(n_row, n_col, i + 1)  # 1-6
        plt.title("Face {}".format(emotion_dict[labels[i]]))
        display_face(images[i], "Face {}".format(emotion_dict[labels[i]]))

def display_subject(subject, layout=(2, 3)):
    (n_row, n_col) = layout
    for i, label in enumerate(subject.label_image_dict.keys()):
        print(subject.label_image_dict)
        plt.subplot(n_row, n_col, i + 1)  # 1-6
        plt.title("{} face".format(label))
        display_face(subject.label_image_dict[label])

def display_eigs(eigen_vectors):
    fake_faces = eigen_vectors.T[0:6]
    for i in range(6):
        plt.subplot(2, 3, i + 1)  # 1-6
        plt.title("eigenface{}".format(i))
        display_face(fake_faces[i].reshape(380, 240), "eigenface {}".format(i + 1))
    plt.show()

# Return the image stored in row wise
def visualize_weights(weights, eig_vecs, labels):
    results = []
    eig_vecs = eig_vecs.T[:weights.shape[1]].T
    print(weights.shape, eig_vecs.shape)
    for weight in weights:
        raw_visualized_weight = np.sum(weight * eig_vecs, axis=1)
        min_val = np.min(raw_visualized_weight)
        max_val = np.max(raw_visualized_weight)
        raw_visualized_weight -= min_val
        raw_visualized_weight /= (max_val - min_val)
        raw_visualized_weight *= 255
        results.append(raw_visualized_weight)

    # for i, result in enumerate(results):
    #     display_face(result, emotion_dict[labels[i]])
    display_faces(results, (2, 3), labels)
    return results