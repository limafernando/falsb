import tensorflow as tf
from tensorflow.keras.initializers import Zeros, Ones

class Encoder2(tf.Module):
    '''
    takes as input x, y and a
    has a hidden layer with 128 neurons
    '''
    def __init__(self, xdim, ydim, adim):
        super().__init__()

        #self.input_output_layer = xdim + ydim + adim
        self.input_output_layer = xdim + ydim
        #self.hidden_layer = 128 #paper implementation
        self.hidden_layer = 256
        #self.shapes = [self.input_output_layer, self.hidden_layer, self.input_output_layer]
        #self.shapes = [self.input_output_layer, self.hidden_layer, self.hidden_layer//2, self.input_output_layer]
        self.shapes = [self.input_output_layer, 256, self.input_output_layer]

        self.zeros = Zeros()
        self.ones = Ones()
        self.initializer = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.5, seed=314159)
        
        self.is_built = False

    def __call__(self, real_data, batch_size):
        
        #batch_size = real_data.shape[0]
        
        if not self.is_built:
            self.Ws = [tf.Variable(self.zeros(shape=(self.shapes[i+1], self.shapes[i])), name='Enc_Ws') 
                                                                                for i in range(len(self.shapes)-1)]
            self.bs = [tf.Variable(self.initializer(shape=(batch_size, self.shapes[i+1])), name='Enc_bs') 
                                                                                for i in range(len(self.shapes)-1)]

            self.is_built = True                                                                                

        prev_layer = real_data
        
        for layer_idx in range(len(self.shapes[1:-1])):

            layer = tf.add(tf.linalg.matmul(prev_layer, tf.transpose(self.Ws[layer_idx])), self.bs[layer_idx])
            #layer = tf.nn.leaky_relu(layer)
            #layer = tf.nn.relu(layer)
            layer = tf.nn.sigmoid(layer)
            prev_layer = layer
        
        layer = tf.add(tf.linalg.matmul(prev_layer, tf.transpose(self.Ws[-1])), self.bs[-1]) #last layer

        #return tf.nn.relu(layer)
        return tf.nn.sigmoid(layer)

class Decoder2(tf.Module):
    '''
    takes as input the 128 neuros from the encoder
    has a hidden layer with 128 neurons
    '''
    def __init__(self, xdim, ydim, adim):
        super().__init__()

        #self.input_output_layer = xdim + ydim + adim
        self.input_output_layer = xdim + ydim
        #self.hidden_layer = 128 #paper implementation
        self.hidden_layer = 256
        #self.shapes = [self.input_output_layer, self.hidden_layer, self.input_output_layer]
        #self.shapes = [self.input_output_layer, self.hidden_layer, self.hidden_layer//2, self.input_output_layer]
        self.shapes = [self.input_output_layer, 256, self.input_output_layer]

        self.zeros = Zeros()
        self.ones = Ones()
        self.initializer = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.5, seed=314159)
        self.is_built = False

    def __call__(self, enc_repr, batch_size):
        
        #batch_size = enc_repr.shape[0]
        
        if not self.is_built:
            self.Ws = [tf.Variable(self.zeros(shape=(self.shapes[i+1], self.shapes[i])), name='Dec_Ws') 
                                                                                for i in range(len(self.shapes)-1)]
            self.bs = [tf.Variable(self.initializer(shape=(batch_size, self.shapes[i+1])), name='Dec_bs') 
                                                                                for i in range(len(self.shapes)-1)]

            self.is_built = True                                                                                

        prev_layer = enc_repr
        
        for layer_idx in range(len(self.shapes[1:-1])):

            layer = tf.add(tf.linalg.matmul(prev_layer, tf.transpose(self.Ws[layer_idx])), self.bs[layer_idx])
            #layer = tf.nn.leaky_relu(layer)
            #layer = tf.nn.relu(layer)
            layer = tf.nn.sigmoid(layer)
            prev_layer = layer
        
        layer = tf.add(tf.linalg.matmul(prev_layer, tf.transpose(self.Ws[-1])), self.bs[-1]) #last layer

        #return tf.nn.relu(layer)
        return tf.nn.sigmoid(layer)

class Autoencoder2(tf.Module):
    '''
    composed by the Encoder and Decoder
    here we must have the get_loss function
    '''
    def __init__(self, xdim, ydim, adim, batch_size):
        super().__init__()

        self.xdim = xdim #input dimensions
        self.ydim = ydim #label dimension
        self.adim = adim #sensitive atribute dimension
        self.batch_size = batch_size

        self.enc = Encoder2(self.xdim, self.ydim, self.adim)
        self.dec = Decoder2(self.xdim, self.ydim, self.adim)

    def __call__(self, X, Y, A):
        #ensure casting
        self.X = tf.dtypes.cast(X, tf.float32)
        self.Y = tf.dtypes.cast(Y, tf.float32)
        self.A = tf.dtypes.cast(A, tf.float32)

        
        self.real_data = tf.concat([self.X, self.Y], 1)
        self.enc_repr = self.enc(self.real_data, self.batch_size)
        self.dec_repr = self.dec(self.enc_repr, self.batch_size)

        self.ae_loss = self.get_ae_loss(self.real_data, self.dec_repr)

    def get_ae_loss(self, real_data, dec_repr):

        loss = tf.square(tf.norm(tensor=(real_data - dec_repr)))

        return loss