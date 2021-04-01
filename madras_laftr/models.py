import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.initializers import GlorotNormal

EPS = 1e-8
CLASS_COEFF = 1.
FAIR_COEFF = 0.
RECON_COEFF = 0.

class Encoder(tf.Module):
    def __init__(self, xdim, hidden_layer_specs, zdim, initializer = GlorotNormal):
        super().__init__()

        self.xdim = xdim #input dimension
        self.hidden_layer_specs = hidden_layer_specs['enc']
        self.zdim = zdim #output dimension
        self.shapes = [self.xdim] + self.hidden_layer_specs + [self.zdim]

        self.ini = initializer()

        self.is_built = False

    def __call__(self, X, hidden_activ_fn=tf.nn.relu):
        
        batch_size = X.shape[0]
        if not self.is_built:
            self.Ws = [tf.Variable(self.ini(shape=(self.shapes[i+1], self.shapes[i])), name='Enc_Ws') 
                                                                                for i in range(len(self.shapes)-1)]
            self.bs = [tf.Variable(self.ini(shape=(batch_size, self.shapes[i+1])), name='Enc_bs') 
                                                                                for i in range(len(self.shapes)-1)]

            self.is_built = True                                                                                
        
        prev_layer = X
             
        for layer_idx in range(len(self.hidden_layer_specs)):
            
            layer = tf.add(tf.linalg.matmul(prev_layer, tf.transpose(self.Ws[layer_idx])), self.bs[layer_idx])
            layer = hidden_activ_fn(layer)
            prev_layer = layer
        
        layer = tf.add(tf.linalg.matmul(prev_layer, tf.transpose(self.Ws[-1])), self.bs[-1]) #last layer

        return layer

class Decoder(tf.Module):
    def __init__(self, zdim, adim, hidden_layer_specs, xdim, initializer = GlorotNormal):
        super().__init__()

        self.z_a_dim = zdim + adim #input dimension
        self.hidden_layer_specs = hidden_layer_specs['enc']
        self.xdim = xdim #output shape
        self.shapes = [self.z_a_dim] + self.hidden_layer_specs + [self.xdim]

        self.ini = initializer()

        self.is_built = False

    def __call__(self, Z_A, hidden_activ_fn=tf.nn.relu):
        
        batch_size = Z_A.shape[0]
        
        if not self.is_built:
            self.Ws = [tf.Variable(self.ini(shape=(self.shapes[i+1], self.shapes[i])), name='Dec_Ws') 
                                                                                for i in range(len(self.shapes)-1)]
            self.bs = [tf.Variable(self.ini(shape=(batch_size, self.shapes[i+1])), name='Dec_bs') 
                                                                                for i in range(len(self.shapes)-1)]

            self.is_built = True

        prev_layer = Z_A
        
        for layer_idx in range(len(self.hidden_layer_specs)):

            layer = tf.add(tf.linalg.matmul(prev_layer, tf.transpose(self.Ws[layer_idx])), self.bs[layer_idx])
            layer = hidden_activ_fn(layer)
            prev_layer = layer
        
        layer = tf.add(tf.linalg.matmul(prev_layer, tf.transpose(self.Ws[-1])), self.bs[-1]) #last layer

        return layer

class Classifier(tf.Module):
    def __init__(self, zdim, hidden_layer_specs, ydim, initializer = GlorotNormal):
        super().__init__()

        self.zdim = zdim #input dimension from latent representation
        self.hidden_layer_specs = hidden_layer_specs['clas']
        self.ydim = ydim #output shape
        self.shapes = [self.zdim] + self.hidden_layer_specs + [self.ydim]

        self.ini = initializer()

        self.is_built = False

    def __call__(self, Z, hidden_activ_fn=tf.nn.relu, out_activ_fn=tf.nn.sigmoid):
        
        batch_size = Z.shape[0]
        if not self.is_built:
            self.Ws = [tf.Variable(self.ini(shape=(self.shapes[i+1], self.shapes[i])), name='Clas_Ws') 
                                                                                for i in range(len(self.shapes)-1)]
            self.bs = [tf.Variable(self.ini(shape=(batch_size, self.shapes[i+1])), name='Clas_bs') 
                                                                                for i in range(len(self.shapes)-1)]

            self.is_built = True                                                                                

        prev_layer = Z
        
        for layer_idx in range(len(self.hidden_layer_specs)):

            layer = tf.add(tf.linalg.matmul(prev_layer, tf.transpose(self.Ws[layer_idx])), self.bs[layer_idx])
            layer = tf.nn.relu(layer)
            prev_layer = layer
        
        layer = tf.add(tf.linalg.matmul(prev_layer, tf.transpose(self.Ws[-1])), self.bs[-1]) #last layer

        return out_activ_fn(layer)

class Adversarial(tf.Module):
    def __init__(self, zdim, hidden_layer_specs, adim, initializer = GlorotNormal):
        super().__init__()

        self.zdim = zdim #input dimension
        self.hidden_layer_specs = hidden_layer_specs['adv']
        self.adim = adim #output shape
        self.shapes = [self.zdim] + self.hidden_layer_specs + [self.adim]

        self.is_built = False

        self.ini = initializer()

    def __call__(self, Z, hidden_activ_fn=tf.nn.relu, out_activ_fn=tf.nn.sigmoid):
        
        batch_size = Z.shape[0]
        if not self.is_built:
            self.Ws = [tf.Variable(self.ini(shape=(self.shapes[i+1], self.shapes[i])), name='Adv_Ws') 
                                                                                for i in range(len(self.shapes)-1)]
            self.bs = [tf.Variable(self.ini(shape=(batch_size, self.shapes[i+1])), name='Adv_bs') 
                                                                                for i in range(len(self.shapes)-1)]

            self.is_built = True                                                                                

        prev_layer = Z
        
        for layer_idx in range(len(self.hidden_layer_specs)):
            
            layer = tf.add(tf.linalg.matmul(prev_layer, tf.transpose(self.Ws[layer_idx])), self.bs[layer_idx])
            layer = hidden_activ_fn(layer)
            prev_layer = layer
        
        layer = tf.add(tf.linalg.matmul(prev_layer, tf.transpose(self.Ws[-1])), self.bs[-1]) #last layer

        return out_activ_fn(layer)

class DemParGan(tf.Module):
    """
    Specialized LAFTR for demographic parity
    """

    def __init__(self, xdim, ydim, adim, zdim, hidden_layer_specs,
                                    recon_coeff=RECON_COEFF, class_coeff=CLASS_COEFF, fair_coeff=FAIR_COEFF):
        super().__init__()

        self.xdim = xdim #input dimensions
        self.ydim = ydim #label dimension
        self.zdim = zdim #reconstruction dimension
        self.adim = adim #sensitive atribute dimension        
        self.hidden_layer_specs = hidden_layer_specs

        self.recon_coeff = recon_coeff
        self.class_coeff = class_coeff
        self.fair_coeff = fair_coeff

        self.enc = Encoder(self.xdim, self.hidden_layer_specs, self.zdim)
        self.clas = Classifier(self.zdim, self.hidden_layer_specs, self.ydim)
        self.adv = Adversarial(self.zdim, self.hidden_layer_specs, self.adim)
        self.dec = Decoder(self.zdim, self.adim, self.hidden_layer_specs, self.xdim)

    def __call__(self, X, Y, A):
        
        #ensure casting
        X = tf.dtypes.cast(X, tf.float32)
        Y = tf.dtypes.cast(Y, tf.float32)
        A = tf.dtypes.cast(A, tf.float32)
        
        self.Z = self.enc(X) #computes the latent representation
        self.Y_hat = self.clas(self.Z, Y) #pred Y
        self.A_hat = self.adv(self.Z) #adversarial prediction for A
        self.X_hat = self.dec(tf.concat([self.Z, self.A_hat], 1)) #reconstructed X

        self.class_loss = self.get_class_loss(self.Y_hat, Y)
        self.recon_loss = self.get_recon_loss(self.X_hat, X)
        self.adv_loss = self.get_advers_loss(self.A_hat, A)
        self.loss = self.get_loss()
        self.class_err = classification_error(Y, self.Y_hat)
        self.adv_err = classification_error(A, self.A_hat)

        return (self.Z, self.Y_hat, self.A_hat, self.X_hat, 
                self.class_loss, self.recon_loss, self.adv_loss, self.loss, self.class_err, self.adv_err)
        
    def get_class_loss(self, Y_hat, Y):
        return cross_entropy(Y, Y_hat)

    def get_recon_loss(self, X_hat, X):
        return tf.reduce_mean(tf.square(X - X_hat), axis=1)

    def get_advers_loss(self, A_hat, A):
        return cross_entropy(A, A_hat)

    def get_loss(self):  # produce losses for the fairness task
        return tf.reduce_mean([
            self.class_coeff*self.class_loss,
            self.recon_coeff*self.recon_loss,
            -self.fair_coeff*self.adv_loss
        ])

class EqOddsUnweightedGan(DemParGan):
    """
    Specialized LAFTR for Equal Odds
    Like DemParGan, but adversarial gets to use the label Y as well
    """

    def __init__(self, xdim, ydim, adim, zdim, hidden_layer_specs):
        super(EqOddsUnweightedGan, self).__init__( xdim, ydim, adim, zdim, hidden_layer_specs)
        
        self.adv = Adversarial(self.zdim + 1 * self.ydim, self.hidden_layer_specs, self.adim)

    def __call__(self, X, Y, A):
        
        self.Z = self.enc(X) #computes the latent representation
        self.Y_hat = self.clas(self.Z, Y) #pred Y
        self.A_hat = self.adv(tf.concat(self.Z, Y)) #adversarial prediction for A
        self.X_hat = self.dec(self.Z) #reconstructed X

        self.class_loss = self.get_class_loss(self.Y_hat, Y)
        self.recon_loss = self.get_recon_loss(self.X_hat, X)
        self.aud_loss = self.get_aud_loss(self.A_hat, A)
        self.loss = self.get_loss()
        self.class_err = classification_error(Y, self.Y_hat)
        self.aud_err = classification_error(A, self.A_hat)

        return self.class_loss, self.recon_loss, self.aud_loss, self.loss, self.class_err, self.aud_err

class EqOppUnweightedGan(DemParGan):
    """
    Specialized LAFTR for Equal Opportunity
    Like DemParGan, but only using Y = 0 examples - this is handled in dataset!
    """

    def __init__(self, xdim, ydim, adim, zdim, hidden_layer_specs):
        super(EqOppUnweightedGan, self).__init__( xdim, ydim, adim, zdim, hidden_layer_specs)

    def get_loss(self):  # produce losses for the fairness task
        loss = self.class_coeff*self.class_loss + self.recon_coeff*self.recon_loss - self.fair_coeff*self.aud_loss
        eqopp_class_loss = tf.multiply(1.  - self.Y, loss)
        return tf.reduce_mean(eqopp_class_loss)

# model-specific utils
def cross_entropy(target, pred, weights=None, eps=EPS):
    if weights == None:
        weights = tf.ones_like(pred)
    return -tf.squeeze(tf.multiply(weights, tf.multiply(target, tf.math.log(pred + eps)) + tf.multiply(1 - target, tf.math.log(1 - pred + eps))))


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