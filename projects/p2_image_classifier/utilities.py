import json
import os
import pathlib

import tensorflow as tf

from makeparse import make_parser
from PIL import Image

IMAGE_SIZE = 224

def print_information(ps=None, class_names=None, flower_names=None):
    if ps is not None:
        print()
        print('The Probabilities of the Flower is :', ps)
        print('The Class of the Flower is : ', class_names)

    if flower_names is not None:
        print('The name of Flower is :', flower_names)


def get_category_names(input_args):

    if not os.path.isfile(input_args):
        raise FileNotFoundError(input_args)

    with open(input_args) as datafile:
        flower_names = json.load(datafile)
    return flower_names


def process_image(image, target_size=(IMAGE_SIZE, IMAGE_SIZE)):

    image_tensor = tf.convert_to_tensor(image)       # Convert the image to a TensorFlow tensor

    resized_image = tf.image.resize(image_tensor,
                                    target_size)     # Resize the image

    normalized_image = resized_image / 255           # Normalize the pixel values to the range [0, 1]

    processed_image = normalized_image.numpy()      # Convert the image back to a Numpy Array

    return processed_image


def get_image_for_prediction(input_args):

    image_path = input_args.input
    print(image_path)
    if pathlib.Path(str(image_path)).exists() is False:
        print('No Path Exists for this Image file')
        exit(1)

    image = Image.open(image_path)

    return image

if __name__ == '__main__':
    parser = make_parser()
    # print(get_category_names())
    parser.input = 'test_images/wild_pansy.jpg'
    image = get_image_for_prediction(parser)
    image_processed = process_image(image)
    print(image_processed)


