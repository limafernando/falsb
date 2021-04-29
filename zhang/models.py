import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.initializers import GlorotNormal
from math import isnan

class Adversarial():
    def __init__(self, initializer = GlorotNormal):
        
        self.ini = initializer()
        #self.U = tf.Variable(1.) #only to initialize
        self.U = tf.Variable(self.ini(shape=(1, 3)), name='U')
        self.is_built = False

    def __call__(self):
        pass

    def build(self, shape):
        return tf.Variable(self.ini(shape=shape), name='U') #check shape

    def get_loss(self, A, A_hat):
        bce = BinaryCrossentropy(from_logits=True)
        return bce(A, A_hat)

class AdversarialDemPar(Adversarial):
    def __init__(self):
        pass

    def __call__(self, Y, Y_hat):
        pass

class AdversarialEqOdds(Adversarial):
    def __init__(self):
        super(AdversarialEqOdds, self).__init__()

        self.c = tf.Variable(self.ini(shape=(1,1)), name='c')

    def __call__(self, Y, Y_hat, b):
        
        #abs_c = 
            
        self.S = tf.math.sigmoid(
                        tf.multiply(
                            (1+tf.math.abs(self.c)), Y_hat#logit(Y_hat)
                        ))

        concatenation = tf.concat(
                    [self.S, tf.multiply(self.S, Y), tf.multiply(self.S, 1-Y)], axis=1
                )

        '''if not self.is_built:
            U_shape = (1, concatenation.shape[1])
            self.U = self.build(U_shape)
            self.is_built = True''' #check how to improve this build of U

        self.A_hat = tf.math.sigmoid(
                        tf.add(
                            tf.matmul(concatenation, tf.transpose(self.U)), 
                            b))

        return self.A_hat

class AdversarialEqOpp(Adversarial):
    def __init__(self):
        pass

    def __call__(self, Y, Y_hat):
        pass

class Classifier():
    def __init__(self, xdim, initializer = GlorotNormal):

        self.ini = initializer()

        self.W = tf.Variable(self.ini(shape=(1, xdim)), name='W')

    def __call__(self, X, b):
        self.Y_hat = tf.math.sigmoid(
                        tf.add(tf.matmul(X, tf.transpose(self.W)), b))

        return self.Y_hat

    def get_loss(self, Y, Y_hat):
        bce = BinaryCrossentropy(from_logits=True)
        return bce(Y, Y_hat+1e-8)

class FairLogisticRegression():
    def __init__(self, xdim, batch_size, fairdef='EqOdds', initializer = GlorotNormal):
        
        self.ini = initializer()
        self.batch_size = batch_size
        
        self.clas = Classifier(xdim)
        adv = self.get_adv_model(fairdef)
        self.adv = adv() #initializing the adversarial object

        self.b = tf.Variable(self.ini(shape=(self.batch_size, 1)), name='b')

    def __call__(self, X, Y, A):

        #ensure casting
        self.X = tf.dtypes.cast(X, tf.float32)
        self.Y = tf.dtypes.cast(Y, tf.float32)
        self.A = tf.dtypes.cast(A, tf.float32)

        self.Y_hat = self.clas(self.X, self.b)
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

########################################################################

def logit(t):
    return tf.subtract(tf.math.log(t), tf.math.log(1-t)) #U are the problem