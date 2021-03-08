from pathlib import Path

def return_npz(data_name):
    data_folder = Path(r'/home/luiz/ufpb/mestrado/code/falsb/data/')
    npz = data_name+'.npz'
    npzfile = data_folder/data_name/'post_prep'/npz
    return npzfile

def return_data_info(data_name):
    dict_keys = ['a0_name', 'a1_name', 'biased', 'even', 'seed', 'use_a']
    
    if data_name == 'adult':
        dict_values = ["Female", "Male", False, False, 2, True]
        return dict(zip(dict_keys, dict_values))