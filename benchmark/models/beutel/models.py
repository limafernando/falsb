import tensorflow as tf
from tensorflow.keras.initializers import GlorotNormal, Zeros, Ones
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy

EPS = 1e-8

######################################################################################################

class Encoder(tf.Module):
    def __init__(self, xdim, hidden_layer_specs, zdim, initializer = GlorotNormal):
        super().__init__()

        self.xdim = xdim #input dimension
        self.hidden_layer_specs = hidden_layer_specs['enc']
        self.zdim = zdim #output dimension
        self.shapes = [self.xdim] + self.hidden_layer_specs + [self.zdim]

        self.ini = initializer()
        self.zeros = Zeros()
        self.ones = Ones()

        self.is_built = False

    def __call__(self, X, hidden_activ_fn=tf.nn.relu):
        
        batch_size = X.shape[0]
        if not self.is_built:
            self.Ws = [tf.Variable(self.zeros(shape=(self.shapes[i+1], self.shapes[i])), name='Enc_Ws') 
                                                                                for i in range(len(self.shapes)-1)]
            self.bs = [tf.Variable(self.ones(shape=(batch_size, self.shapes[i+1])), name='Enc_bs') 
                                                                                for i in range(len(self.shapes)-1)]

            self.is_built = True                                                                                
        
        prev_layer = X
             
        for layer_idx in range(len(self.hidden_layer_specs)):
            
            layer = tf.add(tf.linalg.matmul(prev_layer, tf.transpose(self.Ws[layer_idx])), self.bs[layer_idx])
            layer = tf.nn.leaky_relu(layer)
            prev_layer = layer
        
        layer = tf.add(tf.linalg.matmul(prev_layer, tf.transpose(self.Ws[-1])), self.bs[-1]) #last layer

        return layer

######################################################################################################

class SharedHiddenLayer(tf.Module):
    def __init__(self, ydim, zdim, hidden_layer_specs, initializer = GlorotNormal):
        super().__init__()

        self.zdim = zdim #input dimension from latent representation
        self.hidden_layer_specs = hidden_layer_specs['clas'] #single hl w/ 128
        self.ydim = ydim #output shape
        
        self.shapes = [self.zdim] + self.hidden_layer_specs + [self.ydim]

        self.ini = initializer()
        self.zeros = Zeros()
        self.ones = Ones()

        self.is_built = False

    def __call__(self, Z, hidden_activ_fn=tf.nn.relu, out_activ_fn=tf.nn.sigmoid):
        
        batch_size = Z.shape[0]
        if not self.is_built:
            self.Ws = [tf.Variable(self.zeros(shape=(self.shapes[i+1], self.shapes[i])), name='Clas_Ws') 
                                                                                for i in range(len(self.shapes)-1)]
            self.bs = [tf.Variable(self.ones(shape=(batch_size, self.shapes[i+1])), name='Clas_bs') 
                                                                                for i in range(len(self.shapes)-1)]

            self.is_built = True                                                                                

        prev_layer = Z
        
        for layer_idx in range(len(self.hidden_layer_specs)):

            layer = tf.add(tf.linalg.matmul(prev_layer, tf.transpose(self.Ws[layer_idx])), self.bs[layer_idx])
            layer = tf.nn.relu(layer)
            prev_layer = layer
        
        layer = tf.add(tf.linalg.matmul(prev_layer, tf.transpose(self.Ws[-1])), self.bs[-1])

        return layer #last layer without a activation fn

######################################################################################################

class Classifier(tf.Module):
    def __init__(self, ydim):
        super().__init__()
        
        self.ydim = ydim #output shape

    def __call__(self, logit):

        if self.ydim == 1:
            return tf.nn.sigmoid(logit)
        else:
            return tf.nn.softmax(logit)

######################################################################################################

class Adversarial(tf.Module):
    def __init__(self, adim):
        super().__init__()
        
        self.adim = adim #output shape

    def __call__(self, logit):

        if self.adim == 1:
            return tf.nn.sigmoid(logit)
        else:
            return tf.nn.softmax(logit)

class Beutel(tf.Module):
    """
    Specialized LAFTR for demographic parity
    """

    def __init__(self, xdim, ydim, adim, zdim, hidden_layer_specs, fairdef='DemPar'):
        super().__init__()

        self.xdim = xdim #input dimensions
        self.ydim = ydim #label dimension
        self.zdim = zdim #reconstruction dimension
        self.adim = adim #sensitive atribute dimension        
        self.hidden_layer_specs = hidden_layer_specs
        self.fairdef = fairdef

        self.enc = Encoder(self.xdim, self.hidden_layer_specs, self.zdim)
        self.clas = Classifier(self.ydim, self.zdim, self.hidden_layer_specs)
        self.adv = self.get_adv_model(self.fairdef)

    def __call__(self, X, Y, A):
        
        #ensure casting
        self.X = tf.dtypes.cast(X, tf.float32)
        self.Y = tf.dtypes.cast(Y, tf.float32)
        self.A = tf.dtypes.cast(A, tf.float32)
        
        self.Z = self.enc(self.X) #computes the latent representation
        self.Y_hat = self.clas(self.Z, self.Y) #pred Y

        if self.fairdef == 'EqOpp':
            pass
        
        else:
            self.A_hat = self.adv(self.get_adv_input()) #adversarial prediction for A
        
        self.clas_loss = self.get_clas_loss(self.Y_hat, self.Y, self.ydim)
        self.adv_loss = self.get_advers_loss(self.A_hat, self.A, self.adim)
        
        self.loss = self.get_loss()
        
        self.clas_err = classification_error(self.Y, self.Y_hat)
        self.adv_err = classification_error(self.A, self.A_hat)

        return (self.Z, self.Y_hat, self.A_hat, self.X_hat, 
                self.clas_loss, self.recon_loss, self.adv_loss, self.loss, self.clas_err, self.adv_err)
        
    def get_clas_loss(self, Y_hat, Y):
        return cross_entropy(Y, Y_hat)

    def get_advers_loss(self, A_hat, A, adim):
        return cross_entropy(A, A_hat, adim)

    def get_loss(self):  # produce losses for the fairness task
        return tf.reduce_mean([
            self.clas_coeff*self.clas_loss,
            + self.fair_coeff*self.adv_loss
        ])

    def get_adv_input(self):
        return self.Z

    def get_adv_model(self, fairdef):
        if fairdef == 'DemPar':
            return Adversarial
        # elif fairdef == 'EqOdds':
        #     return AdversarialEqOdds
        elif fairdef == 'EqOpp':
            return Adversarial
        else:
            print('Not a valid fairness definition! Setting to EqOdds!')
            return Adversarial

##############################################################################################

'''model-specific utils'''
def cross_entropy(target, pred, dim=1, weights=None, eps=EPS):
    if dim == 1:
        bce = BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
        return bce(target, pred)
    else: 
        cce = CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
        return cce(target, pred)


def classification_error(target, pred):
    pred_class = tf.round(pred)
    return 1.0 - tf.reduce_mean(tf.cast(tf.equal(target, pred_class), tf.float32))


def wass_loss(target, pred):
    return tf.squeeze(tf.abs(target - pred))


def soft_rate(ind1, ind2, pred): #Y, A, Yhat
    mask = tf.multiply(ind1, ind2)
    rate = tf.reduce_sum(tf.multiply(tf.abs(pred - ind1), mask)) / tf.reduce_sum(mask + EPS)
    return rate


def soft_rate_1(ind1, pred): #Y, A, Yhat
    mask = ind1
    rate = tf.reduce_sum(tf.multiply(pred, mask)) / tf.reduce_sum(mask + EPS)
    return rate