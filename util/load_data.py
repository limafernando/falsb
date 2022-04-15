import numpy as np
import os
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

VALID_DATA_NAMES = ['adult', 'adult-race', 'german', 'titanic', 'heritage-health']
VALID_FILE_NAMES = {
    'adult':'adult', 
    'adult-race':'adult', 
    'german':'german', 
    'titanic':'titanic', 
    'heritage-health':'heritage-health'
}
VALID_LEARNING_STEPS = ['train', 'valid', 'test']
ACCESS_INDEXES = {
    'adult': [slice(-1), -1, -2],
    'adult-race':[slice(-1), -1, slice(63, 68)],
    'german':[],
    'titanic':[],
    'heritage-health':[]
}
DIMENSIONS = {
    'adult':[112, 1, 1], #[X, Y, A]
    'adult-race': [112, 1, 5], 
    'german':'german', 
    'titanic':'titanic', 
    'heritage-health':'heritage-health'
}


def load_data(data_name, learning_step):
    """Function to load data.

    Args:
        data_name (str): used to select the correct data file.
        learning_step (str): used to select the correct data for the learning step.

    Returns:
        [type]: [description]
    """
    if not data_name in VALID_DATA_NAMES:
        print('Invalid data name! Input: {} | Valid data names: [{}]',format(VALID_DATA_NAMES))
        return None

    if not learning_step in VALID_LEARNING_STEPS:
        print('Invalid data name! Input: {} | Valid steps: [{}]',format(VALID_LEARNING_STEPS))
        return None

    data_folder = select_data_folder(data_name)
    access_indexes = get_access_indexes(data_name)
    x, y, a = select_data_step(learning_step, access_indexes, data_folder, data_name)
    
    return x, y, a


def select_data_folder(data_name):
    return os.path.join(ROOT_DIR, Path(r'../data/{}'.format(VALID_FILE_NAMES[data_name])))


def get_access_indexes(data_name):
    return ACCESS_INDEXES[data_name]


def select_data_step(learning_step, access_indexes, data_folder, data_name):
    file = os.path.join(data_folder, Path(r'post_prep/{}.csv'.format(learning_step)))
    data = np.genfromtxt(file, delimiter=',', skip_header=True)[:, 1:]

    num_examples = data.shape[0]
    x = data[:, access_indexes[0]]
    y = data[:, access_indexes[1]].reshape(num_examples, DIMENSIONS[data_name][1])
    a = data[:, access_indexes[2]].reshape(num_examples, DIMENSIONS[data_name][2])
    
    return x, y, a