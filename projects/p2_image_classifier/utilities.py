import json
import os
import pathlib

import tensorflow as tf
from PIL import Image

IMAGE_SIZE = 224

def print_information(ps=None, class_names=None, flower_names=None):
    """
    Prints detailed information about a flower based on provided input.

    Parameters:
    ps: list or None
        A list of probabilities associated with the flower's classification. If not None, it will print the probabilities.

    class_names: list or None
        A list of class labels corresponding to the flower categories. If not None, it will print the class of the flower.

    flower_names: str or None
        A string representing the name of the flower. If not None, it will print the flower's name.
    """
    if ps is not None:
        print()
        print('The Probabilities of the Flower is:', ps)
        print('The Class of the Flower is : ', class_names)

    if flower_names is not None:
        print('The name of Flower is :', flower_names)


def get_category_names(input_args='label_map.json'):
    """
    Reads category names from a JSON file and returns them as a dictionary.

    Parameters:
    input_args (str): The file path to the JSON file containing category names. Defaults to 'label_map.json'.

    Returns:
    dict: A dictionary containing category names.

    Raises:
    FileNotFoundError: If the specified file does not exist.
    """
    if not os.path.isfile(input_args):
        raise FileNotFoundError(input_args)

    with open(input_args) as datafile:
        flower_names = json.load(datafile)
    return flower_names


def process_image(image, target_size=(IMAGE_SIZE, IMAGE_SIZE)):
    """
    Processes the input image by resizing it to the specified target size and normalizing its pixel values.

    Args:
        image: Input image to be processed, expected as a NumPy array or compatible format.
        target_size: Tuple specifying the target size (width, height) to which the image will be resized. Defaults to (IMAGE_SIZE, IMAGE_SIZE).

    Returns:
        A processed image as a NumPy array with normalized pixel values in the range [0, 1].
    """
    image_tensor = tf.convert_to_tensor(image)       # Convert the image to a TensorFlow tensor

    resized_image = tf.image.resize(image_tensor,
                                    target_size)     # Resize the image

    normalized_image = resized_image / 255           # Normalize the pixel values to the range [0, 1]

    processed_image = normalized_image.numpy()      # Convert the image back to a Numpy Array

    return processed_image


def get_image_for_prediction(input_args):
    """
    This function loads an image from the provided file path for further prediction or processing.

    Parameters:
    input_args (Namespace): An object containing input arguments, specifically 'input' which should hold the image file path as a string.

    Returns:
    Image: An Image object loaded from the specified file path.

    Behavior:
    1. Extracts the image file path from the provided input arguments object.
    2. Checks if the specified path exists. If the path does not exist, prints an error message and exits the program.
    3. Opens and loads the image file using the PIL library.

    Raises:
    SystemExit: Exits the program if the provided path does not exist.
    """
    image_path = input_args.input
    if pathlib.Path(str(image_path)).exists() is False:
        print('No Path Exists for this Image file')
        exit(1)

    image = Image.open(image_path)

    return image


