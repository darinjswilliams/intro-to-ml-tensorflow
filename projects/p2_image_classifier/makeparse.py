import argparse

from config import DATA_ROOT


def make_parser():
    parser = argparse.ArgumentParser(description='Command Line Options for Processing Images with TensorFlow')

    parser.add_argument('--dir', type=str, default=DATA_ROOT, help='Directory with images')

    parser.add_argument('--input', type=str, default='test_images', help='Directory with images')

    parser.add_argument('--top_k', dest='top_k', type=int, default=5, help='Return Top k results')

    parser.add_argument('--category_names', dest='category_names', type=str, default='label_map.json', help='Category names')

    parser.add_argument('--checkpoint',  dest='checkpoint', type=str, default='1738342394.h5', help='Default Saved Model')

    parser.add_argument('--saved_dir', dest='saved_dir', type=str, default= 'CheckPoint', help='Directory where models are saved')

    return parser.parse_args()
