import numpy as np
import tensorflow as tf
from PIL import Image

import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

from makeparse import  make_parser
from utilities import (print_information,
                       get_category_names,
                       process_image,
                       get_image_for_prediction)

from model_utils import get_model


def predict(input_args, top_k=5):
    # process image
    image = get_image_for_prediction(input_args)   # return a valid image
    np_image = np.asarray(image)   # convert to numpy array
    class_names = get_category_names(input_args.category_names) # get category name of flowers

    process_return_image = process_image(image=np_image)
    print('Image Return:', process_return_image)

    process_return_image = np.expand_dims(process_return_image, axis=0) #add new axis dimension to array

    model = get_model(input_args)
    print('Model:', model)
    predictions = model.predict(process_return_image)   # make the prediction using the model

    # Get the topk probabilites and class indices
    top_k_probs, top_k_indices = tf.math.top_k(predictions, k=top_k, sorted=True)  # Get topk probabiliites with indices
    top_k_classes = top_k_indices[0].numpy().tolist()  # Get location of indices for reference string name
    flowers = [class_names[str(label)] for label in top_k_classes]  # Get the flower name from class names

    return top_k_probs[0].numpy().tolist(), top_k_indices[0].numpy().tolist(), flowers



def main():

    input_args = make_parser()

    ps, class_name, flower = predict(input_args=input_args,
                                     top_k = input_args.top_k)

    if input_args.category_names:
        print_information(ps=ps,class_names=class_name,flower_names=flower) # print probabilitie, label class name, flower name
    else:
        print_information(ps=ps,class_names=class_name, flower_names=None)   #only print the probabilities and label class names



if __name__ == '__main__':
    main()
    # parser = make_parser()
    # parser.input = 'test_images/wild_pansy.jpg'
    # a, b, c = predict(input_args=parser,
    #                    model=parser.checkpoint,
    #                    top_k=parser.top_k)
    # print(a, b, c)
