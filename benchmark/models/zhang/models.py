import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.initializers import GlorotNormal, RandomNormal
from math import isnan

EPS = 7e-8

class Adversarial():
    def __init__(self, initializer = GlorotNormal):
        
        #self.ini = initializer()
        self.ini = RandomNormal(mean=0.0, stddev=1.5)
        self.U = tf.Variable(1., name='U') #only to initialize
        #self.U = tf.Variable(self.ini(shape=(1, 3)), name='U')
        self.is_built = False
        #self.c = tf.Variable(self.ini(shape=(1,1)), name='c')
        self.c = tf.Variable(tf.ones([1, 1]), name='c')

    def __call__(self):
        pass

    def build(self, shape):
        return tf.Variable(self.ini(shape=shape), name='U') #check shape

    def get_loss(self, A, A_hat):
        A_hat = tf.clip_by_value(A_hat, 1e-9, 1.)
        if self.adim == 1:
            bce = BinaryCrossentropy(from_logits=False)
            return bce(A, A_hat)
        else:
            cce = CategoricalCrossentropy(from_logits=False)
            return cce(A, A_hat)

    '''
    def get_filtered_inputs(self, W, Y, Y_hat, A):
        col = tf.TensorShape([1])
        lines = None
        shape = None
        
        mask = tf.math.equal(Y, 1.)

        #W_filtered = tf.Variable(W[mask])
        #lines = W_filtered.shape
        #shape = col.concatenate(lines)
        #W_filtered = tf.reshape(W_filtered, shape)
        #print('W_fil ', W_filtered.shape)

        Y_filtered = tf.Variable(Y[mask])
        lines = Y_filtered.shape
        shape = lines.concatenate(col)
        Y_filtered = tf.reshape(Y_filtered, shape)

        Y_hat_filtered = tf.Variable(Y_hat[mask])
        lines = Y_hat_filtered.shape
        shape = lines.concatenate(col)
        Y_hat_filtered = tf.reshape(Y_hat_filtered, shape)
        
        A_filtered = tf.Variable(A[mask])
        lines = A_filtered.shape
        shape = lines.concatenate(col)
        A_filtered = tf.reshape(A_filtered, shape)

        return Y_filtered, Y_hat_filtered, A_filtered#, W_filtered
        '''

class AdversarialDemPar(Adversarial):
    def __init__(self, adim):
        super(AdversarialDemPar, self).__init__()
        self.adim = adim
        #self.U = tf.Variable(self.ini(shape=(1, 1)), name='U')
        self.U = tf.Variable(tf.zeros([self.adim, 1]), name='U')

    def __call__(self, Y, Y_hat, b):
        
        self.S = tf.math.sigmoid(
                        tf.multiply(
                            (1+tf.math.abs(self.c)), logit(Y_hat-EPS) #here we add EPS to ensure we wont get a NaN
                        ))

        #self.S = Y_hat

        '''if not self.is_built:
            U_shape = (1, self.S.shape[1])
            self.U = self.build(U_shape)
            #print(self.U)
            self.is_built = True #check how to improve this build of U'''
        if self.adim == 1:
            self.A_hat = tf.math.sigmoid(
                            tf.add(
                                tf.matmul(self.S, tf.transpose(self.U)), 
                                b))
        else:
            self.A_hat = tf.math.softmax(
                            tf.add(
                                tf.matmul(self.S, tf.transpose(self.U)), 
                                b))

        return self.A_hat

class AdversarialEqOdds(Adversarial):
    def __init__(self, adim):
        super(AdversarialEqOdds, self).__init__()
        self.adim = adim
        #self.U = tf.Variable(self.ini(shape=(1, 3)), name='U')
        self.U = tf.Variable(tf.zeros([self.adim, 3]), name='U')

    def __call__(self, Y, Y_hat, b):
        
        self.S = tf.math.sigmoid(
                        tf.multiply(
                            (1+tf.math.abs(self.c)), logit(Y_hat-EPS) #here we add EPS to ensure we wont get a NaN
                        ))
        #print('S ', self.S.shape)
        concatenation = tf.concat(
                    [self.S, tf.multiply(self.S, Y), tf.multiply(self.S, 1-Y)], axis=1
                )
        #print('conc ', concatenation.shape)
        '''if not self.is_built:
            U_shape = (1, concatenation.shape[1])
            self.U = self.build(U_shape)
            #print(self.U)
            self.is_built = True #check how to improve this build of U - getting a warning
        #print('U ', self.U.shape)'''

        if self.adim == 1:
            self.A_hat = tf.math.sigmoid(
                            tf.add(
                                tf.matmul(concatenation, tf.transpose(self.U)), 
                                b))
        else:
            self.A_hat = tf.math.softmax(
                        tf.add(
                            tf.matmul(concatenation, tf.transpose(self.U)), 
                            b))

        
        #print('A_hat ', self.A_hat.shape)
        return self.A_hat

class AdversarialEqOpp(AdversarialEqOdds):
    def __init__(self, adim):
        super(AdversarialEqOpp, self).__init__(adim)

    def filter_As(self, A, A_hat, Y):
        pass
        

##########################################################################################################################

class Classifier():
    def __init__(self, xdim, ydim, initializer = GlorotNormal):
        self.ydim = ydim
        #self.ini = initializer()
        #self.ini = RandomNormal(mean=0.0, stddev=1.5)

        #self.W = tf.Variable(self.ini(shape=(1, xdim)), name='W')
        self.W = tf.Variable(tf.zeros([1, xdim]), name='W')
        #self.W = tf.Variable(tf.ones([1, xdim]), name='W')

    def __call__(self, X, b):
        if self.ydim == 1:
            self.Y_hat = tf.math.sigmoid(
                tf.add(tf.matmul(X, tf.transpose(self.W)), b)
            )

        else:
            self.Y_hat = tf.math.softmax(
                tf.add(tf.matmul(X, tf.transpose(self.W)), b)
            )

        return self.Y_hat

    def get_loss(self, Y, Y_hat):
        Y_hat = tf.clip_by_value(Y_hat, 1e-9, 1.)
        if self.ydim == 1:
            bce = BinaryCrossentropy(from_logits=False)
            return bce(Y, Y_hat)
        else:
            cce = CategoricalCrossentropy(from_logits=False)
            return cce(Y, Y_hat)


##########################################################################################################################

class FairLogisticRegression():
    def __init__(self, xdim, ydim, adim, batch_size, fairdef='EqOdds', initializer = GlorotNormal):
        
        self.ini = initializer()
        self.batch_size = batch_size
        self.fairdef = fairdef
        self.clas = Classifier(xdim, ydim)
        adv = self.get_adv_model(self.fairdef)
        self.adv = adv(adim) #initializing the adversarial object

        self.b = tf.Variable(tf.ones([self.batch_size, 1]), name='b')
        #self.b = tf.Variable(self.ini(shape=(self.batch_size, 1)), name='b')
        #self.b = tf.Variable(tf.zeros([1,1]), name='b')

    def __call__(self, X, Y, A):

        #ensure casting
        self.X = tf.dtypes.cast(X, tf.float32)
        self.Y = tf.dtypes.cast(Y, tf.float32)
        self.A = tf.dtypes.cast(A, tf.float32)

        self.Y_hat = self.clas(self.X, self.b)

        if self.fairdef == 'EqOpp': #trying the filter
            '''
            Y_filtered, Y_hat_filtered, A_filtered = self.adv.get_filtered_inputs(self.clas.W, self.Y, self.Y_hat, self.A)                                                    
            self.A_hat = self.adv(Y_filtered, Y_hat_filtered, self.b)
            print(self.A_hat.shape, A_filtered.shape, Y_filtered.shape, Y_hat_filtered.shape)
            self.adv_loss = self.adv.get_loss(A_filtered, self.A_hat)
            '''
            
            self.A_hat = self.adv(self.Y, self.Y_hat, self.b)

            mask = tf.math.equal(self.Y, 1.).numpy() #to consider only where Y = 1
            mask = mask.reshape(mask.shape[0],)

            col = tf.TensorShape([1])
            lines = None
            shape = None

            A_filtered = tf.boolean_mask(self.A, mask)#tf.Variable(self.A[mask])
            
            lines = A_filtered.shape
            shape = lines.concatenate(col)

            A_filtered = tf.reshape(A_filtered, shape)

            A_hat_filtered = tf.Variable(self.A_hat[mask])

            lines = A_hat_filtered.shape
            shape = lines.concatenate(col)

            A_hat_filtered = tf.reshape(A_hat_filtered, shape)
            
            #self.adv_loss = self.adv.get_loss(A_filtered, A_hat_filtered)
            self.adv_loss = self.adv.get_loss(tf.math.multiply(self.Y, self.A), tf.math.multiply(self.Y, self.A_hat))
            

        else:
            self.A_hat = self.adv(self.Y, self.Y_hat, self.b)
            self.adv_loss = self.adv.get_loss(self.A, self.A_hat)
        
        #self.A_hat = self.adv(self.Y, self.Y_hat, self.b)
        #self.adv_loss = self.adv.get_loss(self.A, self.A_hat)
        
        self.clas_loss = self.clas.get_loss(self.Y, self.Y_hat)
        self.model_loss = self.clas_loss-self.adv_loss

    def get_adv_model(self, fairdef):
        if fairdef == 'DemPar':
            return AdversarialDemPar
        elif fairdef == 'EqOdds':
            return AdversarialEqOdds
        elif fairdef == 'EqOpp':
            return AdversarialEqOpp
        else:
            print('Not a valid fairness definition! Setting to EqOdds!')
            return AdversarialEqOdds

##########################################################################################################################

def logit(t):
    t = tf.math.abs(t)
    return tf.subtract(tf.math.log(t), tf.math.log(1-t))

