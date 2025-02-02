import logging

import numpy as np
import tensorflow as tf

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

from makeparse import  make_parser
from utilities import (print_information,
                       get_category_names,
                       process_image,
                       get_image_for_prediction)

from model_utils import get_model


def predict(input_args, top_k=5):
    """
    Predict the top-k categories and probabilities for a given input using a trained model.

    Parameters:
    input_args: Input arguments or configuration needed for the prediction process.
    top_k (int): The number of top predictions to return. Default is 5.

    Returns:
    tuple: A tuple containing three elements:
        - List of top-k probabilities (floats) for the predictions.
        - List of top-k class indices (integers).
        - List of top-k flower names (strings) corresponding to the predicted classes.

    The function performs the following steps:
    1. Retrieves and processes the input image for prediction.
    2. Converts the image into a required format and adds a new axis.
    3. Loads the model for prediction based on input arguments.
    4. Makes a prediction using the processed image.
    5. Extracts the top-k probabilities and corresponding class indices from the prediction results.
    6. Maps the class indices to flower names using category names.
    """
    image = get_image_for_prediction(input_args)   # return a valid image
    np_image = np.asarray(image)   # convert to numpy array
    class_names = get_category_names() # get category name of flowers

    process_return_image = process_image(image=np_image)
    print('Image Return:', process_return_image)

    process_return_image = np.expand_dims(process_return_image, axis=0) #add new axis dimension to array

    model = get_model(input_args)
    print('Model:', model)
    predictions = model.predict(process_return_image)   # make the prediction using the model


    top_k_probs, top_k_indices = tf.math.top_k(predictions, k=top_k, sorted=True)  # Get topk probabilities with indices
    top_k_classes = top_k_indices[0].numpy().tolist()  # Get location of indices for reference string name
    flowers = [class_names[str(label)] for label in top_k_classes]  # Get the flower name from class names

    return top_k_probs[0].numpy().tolist(), top_k_indices[0].numpy().tolist(), flowers



def main():
    """
    The main function serves as the entry point for executing the script. It manages the flow of operations,
    including parsing input arguments, making predictions, and printing the output information.

    Steps performed by the function:
    1. Input arguments are parsed using `make_parser()` to retrieve necessary parameters.
    2. Predictions are made using the parsed arguments, including retrieving top probabilities, class names, and flower names.
    3. Depending on the presence of the `category_names` argument, it either:
       - Prints the probabilities, label class names, and flower names if `category_names` is provided.
       - Prints only the probabilities and label class names if `category_names` is not provided.
    """
    input_args = make_parser()

    ps, class_name, flower = predict(input_args=input_args,
                                     top_k = input_args.top_k)

    if input_args.category_names == '':
        print(True)
        print_information(ps=ps,class_names=class_name) # print probabilities, label class name, flower name
    else:
        print(False)
        print_information(ps=ps,class_names=class_name, flower_names=flower)   #only print the probabilities and label class names



if __name__ == '__main__':
    main()
