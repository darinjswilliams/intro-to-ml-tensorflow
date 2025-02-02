import os

import tensorflow_hub as hub

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import pathlib

from tensorflow.keras.utils import  custom_object_scope
from tensorflow.keras.models import load_model

from config import DATA_ROOT


def get_model(args):
    """
        Load a pretained model that was saved as h5 format
        :param
            --saved_dir path to where model is saved
            --model_name path name of model
        :return: saved model
    """
    dir_path = os.path.join(str(DATA_ROOT), args.saved_dir)
    file_path = os.path.join(dir_path, args.checkpoint)
    print(file_path)
    if pathlib.Path(str(file_path)).exists() is False:
        print('The Model Path does not exist')
        exit(1)

    with custom_object_scope({'KerasLayer': hub.KerasLayer}):
        reloaded_keras_model = load_model(file_path, compile=False)

    return reloaded_keras_model

