import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.initializers import GlorotNormal

'''class BaseLAFTR(tf.Module):
    def __init__(self, name='BaseLAFTR'):
        super().__init__()


        #COLOCAR OS PARAMETROS AQUI?'''

class Encoder(tf.Module):
    def __init__(self, xdim, hidden_layer_specs, zdim, initializer = GlorotNormal):
        super().__init__()

        self.xdim = xdim #input shape
        self.hidden_layer_specs = hidden_layer_specs['enc']
        self.zdim = zdim #output shape
        self.shapes = [self.xdim] + self.hidden_layer_specs + [self.zdim]

        self.ini = initializer()

        self.Ws = [tf.Variable(self.ini(shape=(self.shapes[i], self.shapes[i+1])), name='Enc_Ws') 
                                                                                for i in range(len(self.shapes)-1)]
        self.bs = [tf.Variable(self.ini(shape=(self.shapes[i], self.shapes[i+1])), name='Enc_bs') 
                                                                                for i in range(len(self.shapes)-1)]

    def __call__(self, X, hidden_activ_fn=tf.nn.relu):
        
        prev_layer = X
        
        for layer_idx in range(len(self.shapes) - 1):
            layer = tf.add(tf.matmul(prev_L, self.Ws[layer_idx]), self.bs[layer_idx])
            layer = hidden_activ_fn(layer)
            prev_layer = layer
        
        layer = tf.add(tf.matmul(prev_L, self.Ws[-1]), self.bs[-1]) #last layer

        return layer

class Decoder(tf.Module):
    def __init__(self, zdim, adim, hidden_layer_specs, xdim, initializer = GlorotNormal):
        super().__init__()

        self.z_a_dim = zdim + adim #input shape
        self.hidden_layer_specs = hidden_layer_specs['enc']
        self.xdim = xdim #output shape
        self.shapes = [self.z_a_dim] + self.hidden_layer_specs + [self.xdim]

        self.ini = initializer()

        self.Ws = [tf.Variable(self.ini(shape=(self.shapes[i], self.shapes[i+1])), name='Dec_Ws') 
                                                                                for i in range(len(self.shapes)-1)]
        self.bs = [tf.Variable(self.ini(shape=(self.shapes[i], self.shapes[i+1])), name='Dec_bs') 
                                                                                for i in range(len(self.shapes)-1)]

    def __call__(self, X, hidden_activ_fn=tf.nn.relu):
        
        prev_layer = X
        
        for layer_idx in range(len(self.shapes) - 1):
            layer = tf.add(tf.matmul(prev_L, self.Ws[layer_idx]), self.bs[layer_idx])
            layer = hidden_activ_fn(layer)
            prev_layer = layer
        
        layer = tf.add(tf.matmul(prev_L, self.Ws[-1]), self.bs[-1]) #last layer

        return layer

class Classifier(tf.Module):
    def __init__(self, zdim, hidden_layer_specs, ydim, initializer = GlorotNormal):
        super().__init__()

        self.zdim = zdim #input shape from latent representation
        self.hidden_layer_specs = hidden_layer_specs['clas']
        self.ydim = ydim #output shape
        self.shapes = [self.zdim] + self.hidden_layer_specs + [self.ydim]

        self.ini = initializer()

        self.Ws = [tf.Variable(self.ini(shape=(self.shapes[i], self.shapes[i+1])), name='Clas_Ws') 
                                                                                for i in range(len(self.shapes)-1)]
        self.bs = [tf.Variable(self.ini(shape=(self.shapes[i], self.shapes[i+1])), name='Clas_bs') 
                                                                                for i in range(len(self.shapes)-1)]

    def __call__(self, X, hidden_activ_fn=tf.nn.relu, out_activ_fn=tf.nn.sigmoid):
        
        prev_layer = X
        
        for layer_idx in range(len(self.shapes) - 1):
            layer = tf.add(tf.matmul(prev_L, self.Ws[layer_idx]), self.bs[layer_idx])
            layer = hidden_activ_fn(layer)
            prev_layer = layer
        
        layer = tf.add(tf.matmul(prev_L, self.Ws[-1]), self.bs[-1]) #last layer

        return out_activ_fn(layer)

class Adversarial(tf.Module):
    def __init__(self, zdim, hidden_layer_specs, adim, initializer = GlorotNormal):
        super().__init__()

        self.zdim = zdim #input shape
        self.hidden_layer_specs = hidden_layer_specs['adv']
        self.adim = adim #output shape
        self.shapes = [self.zdim] + self.hidden_layer_specs + [self.adim]

        self.ini = initializer()

        self.Ws = [tf.Variable(self.ini(shape=(self.shapes[i], self.shapes[i+1])), name='Adv_Ws') 
                                                                                for i in range(len(self.shapes)-1)]
        self.bs = [tf.Variable(self.ini(shape=(self.shapes[i], self.shapes[i+1])), name='Adv_bs') 
                                                                                for i in range(len(self.shapes)-1)]

    def __call__(self, X, hidden_activ_fn=tf.nn.relu, out_activ_fn=tf.nn.sigmoid):
        
        prev_layer = X
        
        for layer_idx in range(len(self.shapes) - 1):
            layer = tf.add(tf.matmul(prev_L, self.Ws[layer_idx]), self.bs[layer_idx])
            layer = hidden_activ_fn(layer)
            prev_layer = layer
        
        layer = tf.add(tf.matmul(prev_L, self.Ws[-1]), self.bs[-1]) #last layer

        return out_activ_fn(layer)

class DemParGan(tf.Module):
    """
    Specialized LAFTR for demographic parity
    """

    def __init__(self, xdim, ydim, adim, zdim, hidden_layer_specs):
        super().__init__()

        self.xdim = xdim #input shape
        self.ydim = ydim #label shape
        self.zdim = zdim #reconstruction shape
        self.adim = adim
        
        self.hidden_layer_specs = hidden_layer_specs
        
        #self.shapes = [self.xdim] + self.hidden_layer_specs + [self.zdim]

        self.enc = Encoder(self.xdim, self.hidden_layer_specs, self.zdim)
        self.clas = Classifier(self.zdim, self.hidden_layer_specs, self.ydim)
        self.adv = Adversarial(self.zdim, self.hidden_layer_specs, self.adim)
        self.dec = Decoder(self.zdim, self.adim, self.hidden_layer_specs, self.xdim)

    def __call__(self, X, Y, A):
        '''
        1: definir Ws e bs
        2: Z = get_latents(X)
        3:
        3.1: Y_hat_logits = class_logits(Z)
        3.2: Y_hat = pred_from_logits(Y_hat_logits) -> aplicar ativação da última camada
        4:
        4.1: A_hat_logits = sensitive_logits(Z)
        4.2: A_hat = pred_from_logits(A_hat_logits) -> aplicar ativação da última camada
        5: X_hat = recon_inputs(Z)
        6: compute losses
        '''

        self.Z = self.enc(X) #computes the latent representation
        self.Y_hat = self.clas(self.Z, Y) #pred Y
        self.A_hat = self.adv(self.Z, Y) #adversarial prediction for A
        self.X_hat = self.dec(self.Z) #reconstructed X

        #losses

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
            -self.fair_coeff*self.advers_loss
        ])
