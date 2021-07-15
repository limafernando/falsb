from pathlib import Path
import numpy as np

def load_data(data_name, learning_step):
    """Function to load data.

    Args:
        data_name (str): used to select the correct data file.
        learning_step (str): used to select the correct data for the learning step.

    Returns:
        [type]: [description]
    """

    data_folder = select_data_folder(data_name)
    x, y, a = select_data_step(learning_step, data_folder)
    
    return x, y, a


def select_data_folder(data_name):
    VALID_DATA_NAMES = 'adult'

    if data_name == 'adult':
        data_folder = Path(r'/home/luiz/ufpb/mestrado/code/falsb/data/adult')
        return data_folder
    else:
        print('Invalid data name! Input: {} | Valid data names: [{}]',format(VALID_DATA_NAMES))



def select_data_step(learning_step, data_folder):
    
    if learning_step == 'train':
        train_file = data_folder/r'post_prep/train.csv'
        train_data = np.genfromtxt(train_file, delimiter=',')

        num_examples = train_data.shape[0]
        x, y, a = train_data[:-10,:-1], train_data[:-10,-1].reshape((num_examples-10, 1)), train_data[:-10,-2].reshape((num_examples-10, 1)) #-10 to do consider only perfect batchs
        #x = train_data[:,:-1]
        #y = train_data[:,-1].reshape(num_examples, 1)
        #a = train_data[:,-2].reshape(num_examples, 1)
        
        return x, y, a
    
    elif learning_step == 'valid':
        valid_file = data_folder/r'post_prep/valid.csv'
        valid_data = np.genfromtxt(valid_file, delimiter=',')

        num_examples = valid_data.shape[0]

        x, y, a = valid_data[:-8,:-1], valid_data[:-8,-1].reshape((num_examples-8, 1)), valid_data[:-8,-2].reshape((num_examples-8, 1))
        #x = valid_data[:,:-1]
        #y = valid_data[:,-1].reshape(num_examples, 1)
        #a = valid_data[:,-2].reshape(num_examples, 1)
        return x, y, a
    
    elif learning_step == 'test':
        test_file = data_folder/r'post_prep/test.csv'
        test_data = np.genfromtxt(test_file, delimiter=',')

        num_examples = test_data.shape[0]

        x, y, a = test_data[:,:-1], test_data[:,-1].reshape((num_examples, 1)), test_data[:,-2].reshape((num_examples, 1))
        #x = test_data[:,:-1]
        #y = test_data[:,-1].reshape(num_examples, 1)
        #a = test_data[:,-2].reshape(num_examples, 1)
        return x, y, a
    
    else:
        print('Invalid learning step! Input: {} | Valid set: [train, test, valid]'.format(learning_step))