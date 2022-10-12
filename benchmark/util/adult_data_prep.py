import numpy as np
from pathlib import Path
from sys import exit
from data_prep_func import *

def main():    
    data_folder = Path(r'/home/luiz/ufpb/mestrado/code/falsb/data/adult')
    
    raw_train = data_folder/'adult.data'
    unprep_test = data_folder/'adult.test'
    unprep_train = data_folder/'adult_train.csv'
    unprep_valid = data_folder/'adult_valid.csv'

    #prep_train = data_folder/r'post_prep/adult.train'
    #prep_test = data_folder/r'post_prep/adult.test'

    #headers_file = data_folder/r'post_prep/adult.headers'

    prep_train = data_folder/r'post_prep/adult_train.csv'
    prep_valid = data_folder/r'post_prep/adult_valid.csv'
    prep_test = data_folder/r'post_prep/adult_test.csv'
    headers_file = data_folder/r'post_prep/adult.headers'
    #data_csv = data_folder/r'post_prep/adult.csv'

    #header_list = open(headers_file, 'w')

    REMOVE_MISSING = True
    MISSING_TOKEN = '?'

    if check_already_prep(headers_file):
        print('Data already processed! Please, check your data files!')
        exit(0)

    headers = 'age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,income'.split(',')
    headers_used = 'age,workclass,education,education-num,marital-status,occupation,relationship,race,capital-gain,capital-loss,hours-per-week,native-country'.split(',')
    target = 'income'
    sensitive = 'sex'

    options = {
        'age': 'buckets',
        'workclass': 'Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked',
        'fnlwgt': 'continuous',
        'education': 'Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool',
        'education-num': 'continuous',
        'marital-status': 'Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse',
        'occupation': 'Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces',
        'relationship': 'Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried',
        'race': 'White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black',
        'sex': 'Female, Male',
        'capital-gain': 'continuous',
        'capital-loss': 'continuous',
        'hours-per-week': 'continuous',
        'native-country': 'United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands',
        'income': ' <=50K,>50K'
    }

    options = {k: [s.strip() for s in sorted(options[k].split(','))] for k in options} #dict values are now sorted lists

    buckets = {'age': [18, 25, 30, 35, 40 ,45, 50, 55, 60, 65]}

    header_prepfunc_map = {
        'age': lambda x: bucket(x, buckets['age']),
        'workclass': lambda x: onehot(x, options['workclass']),
        'fnlwgt': lambda x: continuous(x),
        'education': lambda x: onehot(x, options['education']),
        'education-num': lambda x: continuous(x),
        'marital-status': lambda x: onehot(x, options['marital-status']),
        'occupation': lambda x: onehot(x, options['occupation']),
        'relationship': lambda x: onehot(x, options['relationship']),
        'race': lambda x: onehot(x, options['race']),
        'sex': lambda x: categorical2numerical(x, options['sex']),
        'capital-gain': lambda x: continuous(x),
        'capital-loss': lambda x: continuous(x),
        'hours-per-week': lambda x: continuous(x),
        'native-country': lambda x: onehot(x, options['native-country']),
        'income': lambda x: categorical2numerical(x.strip('.'), options['income']) #strip is needed for test data -> rows ended with '.'
    }

    #start the parse process

    unprep_valid_data, unprep_train_data = set_train_val_split(raw_train)
    save_unprep_data(unprep_valid, unprep_valid_data)
    save_unprep_data(unprep_train, unprep_train_data)

    dataset = {}
    for phase_file, phase in [(unprep_train, 'training'), (unprep_test, 'test'), (unprep_valid, 'validation')]:
        
        unprep_data = [s.strip().split(',') for s in open(phase_file, 'r').readlines()] #s contains the file readlines list

        prep_data = []
        print(phase)

        for line in unprep_data:
            row = [s.strip() for s in line]
            if MISSING_TOKEN in row and REMOVE_MISSING:
                continue
            if row in ([''], ['|1x3 Cross validator']):
                continue
            if row == ['""']:
                continue

            newrow = parse_row(row, headers, headers_used, header_prepfunc_map, sensitive, target)
            prep_data.append(newrow)
            #X.append(newrow)
            #Y.append(label)
            #A.append(sens_att)

        #print(Y)

        dataset[phase] = prep_data

    

    #should write headers file
    save_headers(headers_used+[sensitive]+[target], options, buckets, headers_file)
    files_dict = {'training':prep_train, 'validation':prep_valid, 'test':prep_test}
    success = save_data(files_dict, dataset)

    if success:
        print('Successfully prepared data!')
    else:
        print('Check if data was right prepared!')

#########################################################################################################

def check_already_prep(headers_file):
    #train_file = open(prep_train)
    #test_file = open(prep_test)
    headers_list = open(headers_file)
    return headers_list.readline()

def parse_row(row, headers, headers_used, header_prepfunc_map, sensitive, target):
    new_row_dict = {}
    for i in range(len(row)):
        x = row[i]
        header = headers[i]
        new_row_dict[header] = header_prepfunc_map[header](x)

    sens_att = new_row_dict[sensitive]
    label = new_row_dict[target]
    
    new_row = []
    for h in headers_used:
        new_row = new_row + new_row_dict[h]
    #return new_row, label, sens_att
    return new_row+[sens_att]+[label]

def valid_asserts(data, phase):
    if phase == 'training':
        try:
            assert data.shape == (30162, 114)
        except AssertionError:
            print('Something went wrong with our training data shape!')
            exit('Exit running... Check your data shape!')

    if phase == 'test':
        try:
            assert data.shape == (15060, 114)
        except AssertionError:
            print('Something went wrong with our test data shape!')
            exit('Exit running... Check your data shape!')

#################################################################################################################

if __name__ == "__main__":
    '''
    definir headers, target, sensitive, options e buckets

    chamar main
    '''

    main()