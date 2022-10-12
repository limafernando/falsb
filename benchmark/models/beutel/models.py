import tensorflow as tf
from tensorflow.keras.initializers import GlorotNormal, Zeros, Ones, RandomNormal
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy

EPS = 1e-8

######################################################################################################

class Encoder(tf.Module):
    def __init__(self, xdim, initializer = GlorotNormal):
        super().__init__()
        
        self.xdim = xdim #input dimension
        self.hidden_layer_specs = [128]
        self.zdim = xdim #output dimension equal to xdim
        self.shapes = [self.xdim] + self.hidden_layer_specs + [self.xdim]

        # self.ini = initializer()
        self.ini = RandomNormal(mean=0.5, stddev=0.5)
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
            layer = tf.nn.relu(layer)
            # layer = tf.nn.sigmoid(layer)
            prev_layer = layer
        
        layer = tf.add(tf.linalg.matmul(prev_layer, tf.transpose(self.Ws[-1])), self.bs[-1]) #last layer

        return layer
######################################################################################################

class SharedHiddenLayer(tf.Module):
    def __init__(self, xdim, zdim, initializer = GlorotNormal):
        super().__init__()

        self.xdim = xdim
        self.hidden_layer_specs = [128]
        self.zdim = zdim
        
        # self.shapes = [self.xdim] + self.hidden_layer_specs + [self.zdim]
        self.shapes = [self.xdim] + self.hidden_layer_specs + [1]

        self.ini = initializer()
        self.zeros = Zeros()
        self.ones = Ones()

        self.is_built = False

    def __call__(self, X, hidden_activ_fn=tf.nn.relu, out_activ_fn=tf.nn.sigmoid):
        
        batch_size = X.shape[0]

        if not self.is_built:
            self.Ws = [tf.Variable(self.zeros(shape=(self.shapes[i+1], self.shapes[i])), name='shl_Ws') 
                                                                                for i in range(len(self.shapes)-1)]
            self.bs = [tf.Variable(self.ones(shape=(batch_size, self.shapes[i+1])), name='shl_bs') 
                                                                                for i in range(len(self.shapes)-1)]

            self.is_built = True                                                                                

        prev_layer = X
        
        for layer_idx in range(len(self.hidden_layer_specs)):

            layer = tf.add(tf.linalg.matmul(prev_layer, tf.transpose(self.Ws[layer_idx])), self.bs[layer_idx])
            layer = tf.nn.relu(layer)
            # layer = tf.nn.sigmoid(layer)
            prev_layer = layer
        
        layer = tf.add(tf.linalg.matmul(prev_layer, tf.transpose(self.Ws[-1])), self.bs[-1])

        return layer #last layer without a activation fn

######################################################################################################

class Classifier(tf.Module):
    def __init__(self, ydim, zdim):
        super().__init__()
        
        self.zdim = zdim #output shape
        self.ydim = ydim #output shape
        self.hidden_layer_specs = [128]
        self.shapes = [self.zdim] + self.hidden_layer_specs + [self.ydim]

        # self.ini = initializer()
        self.zeros = Zeros()
        self.ones = Ones()

        self.is_built = False

    def __call__(self, layer, hidden_activ_fn=tf.nn.relu, out_activ_fn=tf.nn.sigmoid):
        
        # batch_size = layer.shape[0]

        # if not self.is_built:
        #     self.Ws = [tf.Variable(self.zeros(shape=(self.shapes[i+1], self.shapes[i])), name='clas_Ws') 
        #                                                                         for i in range(len(self.shapes)-1)]
        #     self.bs = [tf.Variable(self.zeros(shape=(batch_size, self.shapes[i+1])), name='clas_bs') 
        #                                                                         for i in range(len(self.shapes)-1)]

        #     self.is_built = True                                                                                

        # prev_layer = layer
        
        # for layer_idx in range(len(self.hidden_layer_specs)):

        #     layer = tf.add(tf.linalg.matmul(prev_layer, tf.transpose(self.Ws[layer_idx])), self.bs[layer_idx])
        #     layer = tf.nn.relu(layer)
        #     prev_layer = layer
        
        # layer = tf.add(tf.linalg.matmul(prev_layer, tf.transpose(self.Ws[-1])), self.bs[-1])

        if self.ydim == 1:
            return tf.nn.sigmoid(layer)
        else:
            return tf.nn.softmax(layer)

######################################################################################################

class Adversarial(tf.Module):
    def __init__(self, adim, zdim):
        super().__init__()
        
        self.adim = adim #output shape
        self.zdim = zdim #output shape
        self.hidden_layer_specs = [128]
        self.shapes = [self.zdim] + self.hidden_layer_specs + [self.adim]

        # self.ini = initializer()
        self.zeros = Zeros()
        self.ones = Ones()

        self.is_built = False

    def __call__(self, layer, hidden_activ_fn=tf.nn.relu, out_activ_fn=tf.nn.sigmoid):
        
        # batch_size = layer.shape[0]

        # if not self.is_built:
        #     self.Ws = [tf.Variable(self.zeros(shape=(self.shapes[i+1], self.shapes[i])), name='adv_Ws') 
        #                                                                         for i in range(len(self.shapes)-1)]
        #     self.bs = [tf.Variable(self.ones(shape=(batch_size, self.shapes[i+1])), name='adv_bs') 
        #                                                                         for i in range(len(self.shapes)-1)]

        #     self.is_built = True                                                                                

        # prev_layer = layer
        
        # for layer_idx in range(len(self.hidden_layer_specs)):

        #     layer = tf.add(tf.linalg.matmul(prev_layer, tf.transpose(self.Ws[layer_idx])), self.bs[layer_idx])
        #     layer = tf.nn.relu(layer)
        #     prev_layer = layer
        
        # layer = tf.add(tf.linalg.matmul(prev_layer, tf.transpose(self.Ws[-1])), self.bs[-1])

        if self.adim == 1:
            return tf.nn.sigmoid(layer)
        else:
            return tf.nn.softmax(layer)

class Beutel(tf.Module):
    """
    Specialized LAFTR for demographic parity
    """

    def __init__(self, xdim, ydim, adim, zdim, fair_coeff, fairdef='DemPar'):
        super().__init__()

        self.xdim = xdim #input dimensions
        self.ydim = ydim #label dimension
        self.zdim = zdim #reconstruction dimension
        self.adim = adim #sensitive atribute dimension
        self.fair_coeff = fair_coeff
        self.fairdef = fairdef

        self.enc = Encoder(self.xdim)
        self.shl = SharedHiddenLayer(self.xdim, self.zdim)
        self.clas = Classifier(self.ydim, self.zdim)
        self.adv = Adversarial(self.adim, self.zdim)

    def __call__(self, X, Y, A):
        
        # ensure casting
        self.X = tf.dtypes.cast(X, tf.float32)
        self.Y = tf.dtypes.cast(Y, tf.float32)
        self.A = tf.dtypes.cast(A, tf.float32)
        
        self.Z = self.enc(self.X) #computes the latent representation
        # self.Z = self.X
        
        shared_output = self.shl(self.Z)

        self.Y_hat = self.clas(shared_output) #pred Y

        # self.A_hat = self.adv(self.get_adv_input(shared_output)) #pred A
        self.A_hat = self.adv(shared_output*-self.fair_coeff)
        #     tf.multiply(
        #         self.fair_coeff, shared_output
        #     )
        # )

        # if self.fairdef == 'EqOpp':
        #     pass
        
        # else:
        #     self.A_hat = self.adv(self.get_adv_input()) #adversarial prediction for A
        
        self.clas_loss = self.get_clas_loss(self.Y_hat, self.Y, self.ydim)
        self.adv_loss = self.get_advers_loss(self.A_hat, self.A, self.adim)
        # self.adv_loss = -1
        
        self.loss = self.get_loss()
        
        # self.clas_err = classification_error(self.Y, self.Y_hat)
        # self.adv_err = classification_error(self.A, self.A_hat)

        # return (self.Z, self.Y_hat, self.A_hat, self.X_hat, 
        #         self.clas_loss, self.recon_loss, self.adv_loss, self.loss, self.clas_err, self.adv_err)
        
    def get_clas_loss(self, Y_hat, Y, ydim):
        return cross_entropy(Y, Y_hat, ydim)

    def get_advers_loss(self, A_hat, A, adim):
        return cross_entropy(A, A_hat, adim)#*-self.fair_coeff

    def get_loss(self):  # produce losses for the fairness task
        return tf.reduce_mean([
            self.clas_loss,
            self.adv_loss
        ])

    def get_adv_input(self, shared_output):
        # return tf.math.multiply(
        #     -self.fair_coeff, shared_output
        # )
        return shared_output

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