from pathlib import Path
import numpy as np

VALID_DATA_NAMES = ['adult', 'adult-race', 'german', 'titanic', 'heritage-health']
VALID_LEARNING_STEPS = ['train', 'valid', 'test']
ACCESS_INDEXES = {
    'adult': [slice(-1), -1, -2],
    'adult-race':[],
    'german':[],
    'titanic':[],
    'heritage-health':[]
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
    x, y, a = select_data_step(learning_step, access_indexes, data_folder)
    
    return x, y, a


def select_data_folder(data_name):
    return Path(r'data/{}'.format(data_name))


def get_access_indexes(data_name):
    return ACCESS_INDEXES[data_name]


def select_data_step(learning_step, access_indexes, data_folder):
    file = data_folder/r'post_prep/{}.csv'.format(learning_step)
    data = np.genfromtxt(file, delimiter=',', skip_header=True)[:, 1:]

    num_examples = data.shape[0]
    x = data[:, access_indexes[0]]
    y = data[:, access_indexes[1]].reshape(num_examples, 1)
    a = data[:, access_indexes[2]].reshape(num_examples, 1)
    
    return x, y, a