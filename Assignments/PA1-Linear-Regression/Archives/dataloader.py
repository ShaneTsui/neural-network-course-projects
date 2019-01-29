################################################################################
# CSE 253: Programming Assignment 1
# Code snippet by Jenny Hamer
# Winter 2019
################################################################################
# We've provided you with the dataset in CAFE.tar.gz. To uncompress, use:
# tar -xzvf CAFE.tar.gz
################################################################################
# To install PIL, refer to the instructions for your system:
# https://pillow.readthedocs.io/en/5.2.x/installation.html
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################

from os import listdir
from PIL import Image
import numpy as np


# The relative path to your CAFE-Gamma dataset
data_dir = "./CAFE/"

# Dictionary of semantic "label" to emotions
emotion_dict = {"h": "happy", "ht": "happy with teeth", "m": "maudlin",
	"s": "surprise", "f": "fear", "a": "anger", "d": "disgust", "n": "neutral"}


def load_data(data_dir="./CAFE/"):
	""" Load all PGM images stored in your data directory into a list of NumPy
	arrays with a list of corresponding labels.

	Args:
		data_dir: The relative filepath to the CAFE dataset.
	Returns:
		images: A list containing every image in CAFE as an array.
		labels: A list of the corresponding labels (filenames) for each image.
	"""
	# Get the list of image file names
	all_files = listdir(data_dir)

	# Store the images as arrays and their labels in two lists
	images = []
	labels = []

	for file in all_files:
		# Load in the files as PIL images and convert to NumPy arrays
		img = Image.open(data_dir + file)
		images.append(np.array(img))
		labels.append(file)

	print("Total number of images:", len(images), "and labels:", len(labels))

	return images, labels



def display_face(img):
	""" Display the input image and optionally save as a PNG.

	Args:
		img: The NumPy array or image to display

	Returns: None
	"""
	# Convert img to PIL Image object (if it's an ndarray)
	if type(img) == np.ndarray:
		print("Converting from array to PIL Image")
		img = Image.fromarray(img)

	# Display the image
	img.show()
