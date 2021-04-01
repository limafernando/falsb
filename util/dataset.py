import numpy as np
import collections
from util.metrics import *

def mb_round(t, bs):
    """take array t and batch_size bs and trim t to make it divide evenly by bs"""
    new_length = len(t) // bs * bs
    return t[:new_length, :]
 

class Dataset(object):
    def __init__(self, name, a0_name, a1_name, npzfile, seed=0, use_a=False, load_on_init=True, y2i=None, pred_a=False, batch_size=None, **kwargs):
        self.name = name
        self.a0_name = a0_name
        self.a1_name = a1_name
        self.npzfile = npzfile
        self.use_a = use_a
        self.pred_a = pred_a
        self.batch_size = batch_size

        self.loaded = False
        self.seed = seed
        self.y2i = y2i
        if load_on_init:
            self.load()
            self.make_validation_set()

   
    def load(self):
        if not self.loaded:
            data = np.load(self.npzfile)
            self.data = data
            self.x_train = data['x_train']
            print(self.x_train.shape)
            self.x_test = data['x_test'].astype(dtype='float32')
            self.a_train = data['a_train'].astype(dtype='float32')
            self.a_test = data['a_test'].astype(dtype='float32')
            #print('y shape', data['y_train'].shape)
            if data['y_train'].shape[1] > 1:
                #print('changing shape')
                #for y in data['y_train']:
                    #print(y)
                self.y_train = np.expand_dims(data['y_train'][:,1], 1)
                self.y_test = np.expand_dims(data['y_test'][:,1], 1)
            else:
                self.y_train = data['y_train'].astype(dtype='float32')
                self.y_test = data['y_test'].astype(dtype='float32')

            if self.pred_a:
                self.y_train = self.a_train
                self.y_test = self.a_test

            # get valid idxs
            if 'valid_idxs' in data:
                self.train_idxs = data['train_idxs']
                self.valid_idxs = data['valid_idxs']
            if 'y2_train' in data:
                self.y2_train = data['y2_train'].astype(dtype='float32')
                self.y2_test = data['y2_test'].astype(dtype='float32')

            if 'x_valid' in data:
                self.x_valid = data['x_valid'].astype(dtype='float32')
                self.y_valid = np.expand_dims(data['y_valid'][:,1], 1)
                self.a_valid = data['a_valid'].astype(dtype='float32')

            if not self.y2i is None:

                #print('using feature {:d}'.format(self.y2i))
                self.y_train = np.expand_dims(self.y2_train[:,self.y2i], 1)
                self.y_test = np.expand_dims(self.y2_test[:, self.y2i], 1)

            if self.use_a:
                self.x_train = np.concatenate([self.x_train, self.a_train], 1)
                self.x_test = np.concatenate([self.x_test, self.a_test], 1)
                if 'x_valid' in data:
                    self.x_valid = np.concatenate([data['x_valid'], self.a_valid], 1)
            self.loaded = True

    def make_validation_set(self, force=False):
        if not hasattr(self, 'x_valid') or force:
            self.x_valid = self.x_train[self.valid_idxs]
            self.y_valid = self.y_train[self.valid_idxs]
            self.a_valid = self.a_train[self.valid_idxs]

            self.x_train = self.x_train[self.train_idxs]
            print('mk val set', self.x_train.shape)
            self.y_train = self.y_train[self.train_idxs]
            self.a_train = self.a_train[self.train_idxs]
            
            if hasattr(self, 'y2_valid'):
                self.y2_valid = self.y2_train[self.valid_idxs]
                self.y2_train = self.y2_train[self.train_idxs]
        
        # hack for WGAN-GP training: trim to bacth size if a batch size is specified
        if self.batch_size is not None:
            self.x_train = mb_round(self.x_train, self.batch_size)
            self.x_test = mb_round(self.x_test, self.batch_size)
            self.x_valid = mb_round(self.x_valid, self.batch_size)
            self.a_train = mb_round(self.a_train, self.batch_size)
            self.a_test = mb_round(self.a_test, self.batch_size)
            self.a_valid = mb_round(self.a_valid, self.batch_size)
            self.y_train = mb_round(self.y_train, self.batch_size)
            self.y_test = mb_round(self.y_test, self.batch_size)
            self.y_valid = mb_round(self.y_valid, self.batch_size)
            
            if hasattr(self, 'y2_train'):
                self.y2_train = mb_round(self.y2_train, self.batch_size)
                self.y2_test = mb_round(self.y2_test, self.batch_size)
                self.y2_valid = mb_round(self.y2_valid, self.batch_size)

    def get_A_proportions(self):
        A0 = NR(self.a_train)
        A1 = PR(self.a_train)
        assert A0 + A1 == 1
        return [A0, A1]

    def get_Y_proportions(self):
        Y0 = NR(self.y_train)
        Y1 = PR(self.y_train)
        assert Y0 + Y1 == 1
        return [Y0, Y1]

    def get_AY_proportions(self):
        ttl = float(self.y_train.shape[0])
        A0Y0 = TN(self.y_train, self.a_train) / ttl
        A0Y1 = FN(self.y_train, self.a_train) / ttl
        A1Y0 = FP(self.y_train, self.a_train) / ttl
        A1Y1 = TP(self.y_train, self.a_train) / ttl
        return [[A0Y0, A0Y1], [A1Y0, A1Y1]]

    def get_batch_iterator(self, phase, mb_size):
        if phase == 'train':
            x = self.x_train
            y = self.y_train
            a = self.a_train
        elif phase == 'valid':
            x = self.x_valid
            y = self.y_valid
            a = self.a_valid
        elif phase == 'test':
            x = self.x_test
            y = self.y_test
            a = self.a_test
        else:
            raise Exception("invalid phase name")

        size = x.shape[0]
        batch_idxs = make_batch_idxs(size, mb_size, self.seed, phase)
        iterator = DatasetIterator([x, y, a], batch_idxs)
        return iterator

    def get_shapes(self):
        x_train = self.x_train.shape
        print(x_train)
        y_train = self.y_train.shape
        a_train = self.a_train.shape
        x_test = self.x_test.shape
        y_test = self.y_test.shape
        a_test = self.a_test.shape
        x_valid = self.x_valid.shape
        y_valid = self.y_valid.shape
        a_valid = self.a_valid.shape
        return x_train, y_train, a_train, x_test, y_test, a_test, x_valid, y_valid, a_valid

class TransferDataset(Dataset):
    def __init__(self, reprs, A, label_index, Y_loaded=None, phase='Test', **data_kwargs):
        super().__init__(**data_kwargs)
        if label_index == 'a':
            Y = A
        elif label_index >= 0:
            Y2 = self.y2_test if phase == 'Test' else self.y2_valid
            Y = np.expand_dims(Y2[:,label_index], 1)
        else:
            assert not Y_loaded is None
            Y = Y_loaded
            assert np.array_equal(Y, self.y_test) or np.array_equal(Y, self.y_valid)
        assert Y.shape[0] == reprs.shape[0]
        x_train, x_test, y_train, y_test, a_train, a_test = self.make_train_test_split(reprs, A, Y)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.a_train = a_train
        self.a_test = a_test
        self.train_idxs, self.valid_idxs = self.make_valid_idxs(x_train, pct=0.2)
        self.make_validation_set(force=True)

    def make_valid_idxs(self, X, pct):
        np.random.seed(self.seed)

        n = X.shape[0]
        shuf = np.arange(n)
        valid_pct = pct
        valid_ct = int(n * valid_pct)
        valid_idxs = shuf[:valid_ct]
        train_idxs = shuf[valid_ct:]

        return train_idxs, valid_idxs

    def make_train_test_split(self, X, A, Y):
        #print(X.shape, A.shape, Y.shape)
        tr_idxs, te_idxs = self.make_valid_idxs(X, pct=0.3)
        X_tr = X[tr_idxs,:]
        X_te = X[te_idxs,:]
        Y_tr = Y[tr_idxs,:]
        Y_te = Y[te_idxs,:]
        A_tr = A[tr_idxs,:]
        A_te = A[te_idxs,:]
        return X_tr, X_te, Y_tr, Y_te, A_tr, A_te


class DatasetIterator(collections.Iterator):
    def __init__(self, tensor_list, idxs_list):
        self.tensors = tensor_list
        self.idxs = idxs_list
        self.curr = 0
        self.minibatches = len(self.idxs)

    def __iter__(self):
        return self

    def __next__(self):
        if self.curr >= self.minibatches:
            raise StopIteration
        else:
            idxs = self.idxs[self.curr]
            minibatch = [tensor[idxs] for tensor in self.tensors]
            self.curr += 1
            return minibatch #a minibatch is a list of tensor's slices


def make_batch_idxs(size, mb_size, seed=0, phase='train'):
    np.random.seed(seed)
    
    if phase == 'train': #firstly, make a shuffle
        shuf = np.random.permutation(size)
    else:
        shuf = np.arange(size)
    
    start = 0
    mbs = []
    while start < size:
        end = min(start + mb_size, size)
        mb_i = shuf[start:end]
        mbs.append(mb_i)
        start = end
    return mbs
