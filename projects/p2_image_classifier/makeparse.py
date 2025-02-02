import argparse

from config import DATA_ROOT


def make_parser():
    """
    Creates an argparse.ArgumentParser object and configures command-line arguments for processing images with TensorFlow.

    Returns:
        argparse.Namespace: Parsed argument object containing the values provided through command-line arguments.

    Arguments:
        --dir: Specifies the directory containing images. Defaults to the predefined DATA_ROOT.
        --input: Specifies the input image directory. Defaults to 'test_images'.
        --top_k: Determines the number of top results to return. Defaults to 5.
        --category_names: Specifies the category names file. Defaults to an empty string.
        --checkpoint: Specifies the saved model checkpoint to use. Defaults to '1738342394.h5'.
        --saved_dir: Specifies the directory where models are saved. Defaults to 'CheckPoint'.
    """
    parser = argparse.ArgumentParser(description='Command Line Options for Processing Images with TensorFlow')

    parser.add_argument('--dir', type=str, default=DATA_ROOT, help='Directory with images')

    parser.add_argument('--input', type=str, default='test_images', help='Directory with images')

    parser.add_argument('--top_k', dest='top_k', type=int, default=5, help='Return Top k results')

    parser.add_argument('--category_names',type=str, default='', help='Category names')

    parser.add_argument('--checkpoint',  dest='checkpoint', type=str, default='1738342394.h5', help='Default Saved Model')

    parser.add_argument('--saved_dir', dest='saved_dir', type=str, default= 'CheckPoint', help='Directory where models are saved')

    return parser.parse_args()
