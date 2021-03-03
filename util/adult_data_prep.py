import numpy as np
from pathlib import Path
from sys import exit

def main():    
    data_folder = Path(r'/home/luiz/ufpb/mestrado/code/falsb/data/adult_data')
    
    unprep_train = data_folder/'adult.data'
    unprep_test = data_folder/'adult.test'

    #prep_train = data_folder/r'post_prep/adult.train'
    #prep_test = data_folder/r'post_prep/adult.test'

    #headers_file = data_folder/r'post_prep/adult.headers'

    data_npz = data_folder/r'post_prep/adult.npz'
    headers_file = data_folder/r'post_prep/adult.headers'
    data_csv = data_folder/r'post_prep/adult.csv'

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

    options = {k: [s.strip() for s in sorted(options[k].split(','))] for k in options}

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
        'sex': lambda x: onehot(x, options['sex']),
        'capital-gain': lambda x: continuous(x),
        'capital-loss': lambda x: continuous(x),
        'hours-per-week': lambda x: continuous(x),
        'native-country': lambda x: onehot(x, options['native-country']),
        'income': lambda x: onehot(x.strip('.'), options['income']),
    }

    dataset = {}
    for phase_file, phase in [(unprep_train, 'training'), (unprep_test, 'test')]:
        unprep_data = [s.strip().split(',') for s in open(phase_file, 'r').readlines()]

        X = []
        Y = []
        A = []
        print(phase)

        for line in unprep_data:
            row = [s.strip() for s in line]
            if MISSING_TOKEN in row and REMOVE_MISSING:
                continue
            if row in ([''], ['|1x3 Cross validator']):
                continue
            newrow, label, sens_att = parse_row(row, headers, headers_used, header_prepfunc_map, sensitive, target)
            X.append(newrow)
            Y.append(label)
            A.append(sens_att)

        npX = np.array(X)
        npY = np.array(Y)
        npA = np.array(A)
        npA = np.expand_dims(npA[:,1], 1)

        dataset[phase] = {}
        dataset[phase]['X'] = npX
        dataset[phase]['Y'] = npY
        dataset[phase]['A'] = npA

        valid_asserts(npX, npY, npA, phase)

    #should write headers file
    headers_list = open(headers_file, 'w')
    i = 0
    for h in headers_used:
        if options[h] == 'continuous':
            headers_list.write('{:d},{}\n'.format(i, h))
            i += 1
        elif options[h][0] == 'buckets':
            for b in buckets[h]:
                colname = '{}_{:d}'.format(h, b)
                headers_list.write('{:d},{}\n'.format(i, colname))
                i += 1
        else:
            for opt in options[h]:
                colname = '{}_{}'.format(h, opt)
                headers_list.write('{:d},{}\n'.format(i, colname))
                i += 1

    valid_idxs, train_idxs = set_train_val_idxs(dataset)

    np.savez(data_npz, x_train=dataset['training']['X'], x_test=dataset['test']['X'],
                y_train=dataset['training']['Y'], y_test=dataset['test']['Y'],
                a_train=dataset['training']['A'], a_test=dataset['test']['A'],
                train_idxs=train_idxs, valid_inds=valid_idxs)

    print('Successfully prepared data!')

def check_already_prep(headers_file):
    #train_file = open(prep_train)
    #test_file = open(prep_test)
    headers_list = open(headers_file)
    return headers_list.readline()

def parse_row(row, headers, headers_used, hpm, sensitive, target):
    new_row_dict = {}
    for i in range(len(row)):
        x = row[i]
        header = headers[i]
        new_row_dict[header] = hpm[header](x)
    sens_att = new_row_dict[sensitive]
    label = new_row_dict[target]
    new_row = []
    for h in headers_used:
        new_row = new_row + new_row_dict[h]
    return new_row, label, sens_att

def valid_asserts(npX, npY, npA, phase):
    if phase == 'training':
        try:
            assert npX.shape == (30162, 112)
            assert npY.shape == (30162, 2)
            assert npA.shape == (30162, 1)
        except AssertionError:
            print('Something went wrong with our training data shape!')
            exit('Exit running... Check your data shape!')

    if phase == 'test':
        try:
            assert npX.shape == (15060, 112)
            assert npY.shape == (15060, 2)
            assert npA.shape == (15060, 1)
        except AssertionError:
            print('Something went wrong with our test data shape!')
            exit('Exit running... Check your data shape!')

def bucket(x, buckets):
    x = float(x)
    n = len(buckets)
    label = n
    for i in range(n):
        if x <= buckets[i]:
            label = i
            break
    template = [0. for j in range(n + 1)]
    template[label] = 1.
    return template

def onehot(x, choices):
    if not x in choices:
        print('could not find "{}" in choices'.format(x))
        print(choices)
        raise Exception()
    label = choices.index(x)
    template = [0. for j in range(len(choices))]
    template[label] = 1.
    return template

def continuous(x):
    return [float(x)]

def normalization(dataset):
    mean = np.mean(dataset['training']['X'], axis=0)
    std = np.std(dataset['training']['X'], axis=0)
    print(mean, std)
    dataset['training']['X'] = whiten(dataset['training']['X'], mean, std)
    dataset['test']['X'] = whiten(dataset['test']['X'], mean, std)

def whiten(X, mean, std):
    EPS = 1e-8
    meantile = np.tile(mean, (X.shape[0], 1))
    stdtile = np.maximum(np.tile(std, (X.shape[0], 1)), EPS)
    X = X - meantile
    X = np.divide(X, stdtile)
    return X

def set_train_val_idxs(dataset):
    train_x = dataset['training']['X'].shape[0]
    shuf = np.random.permutation(train_x)
    valid_pct = 0.2
    valid_ct = int(train_x * valid_pct)
    valid_idxs = shuf[:valid_ct]
    train_idxs = shuf[valid_ct:]
    return valid_idxs, train_idxs

if __name__ == "__main__":
    '''
    definir headers, target, sensitive, options e buckets

    chamar main
    '''

    main()