import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.initializers import GlorotNormal, RandomNormal
from math import isnan

EPS = 7e-8

class Adversarial():
    def __init__(self, initializer = GlorotNormal):
        
        self.ini = initializer()
        self.U = tf.Variable(1., name='U') #only to initialize
        #self.U = tf.Variable(self.ini(shape=(1, 3)), name='U')
        self.is_built = False
        self.c = tf.Variable(self.ini(shape=(1,1)), name='c')

    def __call__(self):
        pass

    def build(self, shape):
        return tf.Variable(self.ini(shape=shape), name='U') #check shape

    def get_loss(self, A, A_hat):
        bce = BinaryCrossentropy(from_logits=True)
        return bce(A, A_hat)

    def get_filtered_inputs(self, W, Y, Y_hat, A):
        col = tf.TensorShape([1])
        lines = None
        shape = None
        
        mask = tf.math.equal(Y, 1.)

        '''W_filtered = tf.Variable(W[mask])
        lines = W_filtered.shape
        shape = col.concatenate(lines)
        W_filtered = tf.reshape(W_filtered, shape)
        print('W_fil ', W_filtered.shape)'''

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

class AdversarialDemPar(Adversarial):
    def __init__(self):
        super(AdversarialDemPar , self).__init__()

    def __call__(self, Y, Y_hat, b):
        
        #abs_c = 
            
        self.S = tf.math.sigmoid(
                        tf.multiply(
                            (1+tf.math.abs(self.c)), logit(Y_hat-EPS) #here we add EPS to ensure we wont get a NaN
                        ))

        if not self.is_built:
            U_shape = (1, self.S.shape[1])
            self.U = self.build(U_shape)
            #print(self.U)
            self.is_built = True #check how to improve this build of U

        self.A_hat = tf.math.sigmoid(
                        tf.add(
                            tf.matmul(self.S, tf.transpose(self.U)), 
                            b))

        return self.A_hat

class AdversarialEqOdds(Adversarial):
    def __init__(self):
        super(AdversarialEqOdds, self).__init__()

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
        if not self.is_built:
            U_shape = (1, concatenation.shape[1])
            self.U = self.build(U_shape)
            #print(self.U)
            self.is_built = True #check how to improve this build of U - getting a warning
        #print('U ', self.U.shape)
        self.A_hat = tf.math.sigmoid(
                        tf.add(
                            tf.matmul(concatenation, tf.transpose(self.U)), 
                            b))
        #print('A_hat ', self.A_hat.shape)
        return self.A_hat

class AdversarialEqOpp(AdversarialEqOdds):
    def __init__(self):
        super(AdversarialEqOpp, self).__init__()

##########################################################################################################################

class Classifier():
    def __init__(self, xdim, initializer = GlorotNormal):

        #self.ini = initializer()
        self.ini = RandomNormal(mean=0.0, stddev=1.5)

        self.W = tf.Variable(self.ini(shape=(1, xdim)), name='W')
        #self.W = tf.Variable(tf.ones([1, xdim]), name='W')

    def __call__(self, X, b):
        self.Y_hat = tf.math.sigmoid(
                        tf.add(tf.matmul(X, tf.transpose(self.W)), b))

        return self.Y_hat

    def get_loss(self, Y, Y_hat):
        bce = BinaryCrossentropy(from_logits=True)
        return bce(Y, Y_hat)

##########################################################################################################################

class FairLogisticRegression():
    def __init__(self, xdim, batch_size, fairdef='EqOdds', initializer = GlorotNormal):
        
        self.ini = initializer()
        self.batch_size = batch_size
        self.fairdef = fairdef
        self.clas = Classifier(xdim)
        adv = self.get_adv_model(self.fairdef)
        self.adv = adv() #initializing the adversarial object

        #self.b = tf.Variable(self.ini(shape=(self.batch_size, 1)), name='b')
        self.b = tf.Variable(tf.zeros([1]), name='b')

    def __call__(self, X, Y, A):

        #ensure casting
        self.X = tf.dtypes.cast(X, tf.float32)
        self.Y = tf.dtypes.cast(Y, tf.float32)
        self.A = tf.dtypes.cast(A, tf.float32)

        self.Y_hat = self.clas(self.X, self.b)

        '''if self.fairdef == 'EqOpp': #trying the filter
            #Y_filtered, Y_hat_filtered, A_filtered, self.W_filtered = self.adv.get_filtered_inputs(self.clas.W, self.Y, self.Y_hat, self.A)
            Y_filtered, Y_hat_filtered, A_filtered = self.adv.get_filtered_inputs(self.clas.W,
                                                                        self.Y, self.Y_hat, self.A)                                                    
            self.A_hat = self.adv(Y_filtered, Y_hat_filtered, self.b)
            print(self.A_hat.shape, A_filtered.shape, Y_filtered.shape, Y_hat_filtered.shape)
            self.adv_loss = self.adv.get_loss(A_filtered, self.A_hat)

        else:
            self.A_hat = self.adv(self.Y, self.Y_hat, self.b)
            self.adv_loss = self.adv.get_loss(self.A, self.A_hat)'''
        
        self.A_hat = self.adv(self.Y, self.Y_hat, self.b)
        self.adv_loss = self.adv.get_loss(self.A, self.A_hat)
        
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

