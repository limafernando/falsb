import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.initializers import GlorotNormal, Zeros, Ones
from tensorflow.keras.losses import BinaryCrossentropy

class Encoder(tf.Module):
    '''
    takes as input x, y and a
    has a hidden layer with 128 neurons
    '''
    def __init__(self, xdim, ydim, adim):
        super().__init__()

        self.input_output_layer = xdim + ydim + adim
        self.hidden_layer = 128
        self.shapes = [self.input_output_layer, self.hidden_layer, self.input_output_layer]

        self.zeros = Zeros()
        self.ones = Ones()
        
        self.is_built = False

    def __call__(self, real_data):
        
        batch_size = real_data.shape[0]
        
        if not self.is_built:
            self.Ws = [tf.Variable(self.zeros(shape=(self.shapes[i+1], self.shapes[i])), name='Enc_Ws') 
                                                                                for i in range(len(self.shapes)-1)]
            self.bs = [tf.Variable(self.ones(shape=(batch_size, self.shapes[i+1])), name='Enc_bs') 
                                                                                for i in range(len(self.shapes)-1)]

            self.is_built = True                                                                                

        prev_layer = real_data
        
        for layer_idx in range(len(self.shapes[1:-1])):

            layer = tf.add(tf.linalg.matmul(prev_layer, tf.transpose(self.Ws[layer_idx])), self.bs[layer_idx])
            layer = tf.nn.leaky_relu(layer)
            prev_layer = layer
        
        layer = tf.add(tf.linalg.matmul(prev_layer, tf.transpose(self.Ws[-1])), self.bs[-1]) #last layer

        return tf.nn.sigmoid(layer)

class Decoder(tf.Module):
    '''
    takes as input the 128 neuros from the encoder
    has a hidden layer with 128 neurons
    '''
    def __init__(self, xdim, ydim, adim):
        super().__init__()

        self.input_output_layer = xdim + ydim + adim
        self.hidden_layer = 128
        self.shapes = [self.input_output_layer, self.hidden_layer, self.input_output_layer]

        self.zeros = Zeros()
        self.ones = Ones()
        
        self.is_built = False

    def __call__(self, enc_repr):
        
        batch_size = enc_repr.shape[0]
        
        if not self.is_built:
            self.Ws = [tf.Variable(self.zeros(shape=(self.shapes[i+1], self.shapes[i])), name='Dec_Ws') 
                                                                                for i in range(len(self.shapes)-1)]
            self.bs = [tf.Variable(self.ones(shape=(batch_size, self.shapes[i+1])), name='Dec_bs') 
                                                                                for i in range(len(self.shapes)-1)]

            self.is_built = True                                                                                

        prev_layer = enc_repr
        
        for layer_idx in range(len(self.shapes[1:-1])):

            layer = tf.add(tf.linalg.matmul(prev_layer, tf.transpose(self.Ws[layer_idx])), self.bs[layer_idx])
            layer = tf.nn.leaky_relu(layer)
            prev_layer = layer
        
        layer = tf.add(tf.linalg.matmul(prev_layer, tf.transpose(self.Ws[-1])), self.bs[-1]) #last layer

        return tf.nn.sigmoid(layer)

class Autoencoder():
    '''
    composed by the Encoder and Decoder
    here we must have the get_loss function
    '''
    def __init__(self, xdim, ydim, adim):
        self.xdim = xdim #input dimensions
        self.ydim = ydim #label dimension
        self.adim = adim #sensitive atribute dimension

        self.enc = Encoder(self.xdim, self.ydim, self.adim)
        self.dec = Decoder(self.xdim, self.ydim, self.adim)

    def __call__(self, X, Y, A):
        #ensure casting
        self.X = tf.dtypes.cast(X, tf.float32)
        self.Y = tf.dtypes.cast(Y, tf.float32)
        self.A = tf.dtypes.cast(A, tf.float32)

        self.real_data = tf.concat([self.X, self.Y, self.A], 1)

        self.enc_repr = self.enc(self.real_data)
        self.dec_repr = self.dec(self.enc_repr)

        self.loss = self.get_loss(self.real_data, self.dec_repr)

    def get_loss(self, real_data, dec_repr):
        '''euclidian distance'''
        #dec_repr - tf.concat(X, Y, A)
        loss = tf.sqrt(
                tf.math.reduce_sum(
                    tf.math.squared_difference(real_data, dec_repr)
                )
        )

        return loss


######################################

class Generator():
    #take as input the protected attribute (a) and the noise variable (z)
    pass

class GDec():
    pass #GDec = Dec(G(z))

class Discriminator():
    pass

class FairDiscriminator():
    pass