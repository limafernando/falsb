from pathlib import Path
import os

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

def make_dir_if_not_exist(d, remove=False):
    os.chdir('..')
    print(d)
    if remove and os.path.exists(d):  # USE AT YOUR OWN RISK
        import shutil
        shutil.rmtree(d)
    if not os.path.exists(d):
        os.makedirs(d)