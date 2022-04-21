import numpy as np
from csv import writer

def continuous(x):
    return [float(x)]

def categorical2numerical(x, choices):
    label = choices.index(x)
    return label

def bucket(x, buckets):
    x = float(x)
    n = len(buckets)
    label = n
    for i in range(n):
        if x <= buckets[i]:
            label = i
            break
    # template = [0. for j in range(n + 1)]
    # template[label] = 1.
    # return template
    return label

def onehot(x, choices):
    if not x in choices:
        print('could not find "{}" in choices'.format(x))
        print(choices)
        raise Exception()
    label = choices.index(x)
    template = [0. for j in range(len(choices))]
    template[label] = 1.
    return template

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

def set_train_val_split(train_file, valid_pct=0.2):
    unprep_data = [s.strip().split(',') for s in open(train_file, 'r').readlines()]
    np_data = np.array(unprep_data)
    len_data = np_data.shape[0]
    shuf = np.random.permutation(np_data) #random shuffle
    valid_ct = int(len_data * valid_pct)
    valid_data = shuf[:valid_ct]
    train_data = shuf[valid_ct:]
    return valid_data.tolist(), train_data.tolist()

def save_headers(headers, options, buckets, headers_file):
    headers_list = open(headers_file, 'w')
    i = 0

    for h in headers:
        if options[h] == 'continuous' or len(options[h]) == 2:
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

def save_data(files_dict, dataset, phases=['training', 'validation', 'test']):
    for phase in phases:
        with open(files_dict[phase], 'w') as csvfile:
            w = writer(csvfile, delimiter=",")
            
            data = dataset[phase]
            for line in data:
                w.writerow(line)
        
        csvfile.close()
    
    return True #eveything is okay

def save_unprep_data(file, data):
    
    REMOVE_MISSING = True
    MISSING_TOKEN = '?'
    
    with open(file, 'w') as csvfile:
        w = writer(csvfile, delimiter=",")
        
        for line in data:
            line = [s.strip() for s in line]
            if MISSING_TOKEN in line and REMOVE_MISSING:
                continue
            if line in ([''], ['|1x3 Cross validator']):
                continue
            w.writerow(line)
        
    csvfile.close()