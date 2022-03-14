import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.initializers import GlorotNormal, Zeros, Ones
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.python.ops.gen_batch_ops import batch

class Generator2(tf.Module):
    #takes as input the protected attribute (a) and the noise variable (z)

    def __init__(self, xdim, ydim, adim, noise_dim, dec):
        super().__init__()
        '''self.zdim = xdim + ydim #in our dataset the protected attribute is included in the att vector x
        self.adim = adim'''
        self.zdim = xdim + ydim #random noise dimension
        self.data_dim = xdim + adim + ydim
        self.noise_dim = noise_dim
        self.dec = dec
        #self.hidden_layer = [128,128] #paper implementation
        #self.shapes = [self.data_dim, 128, 256, 512, 512, 256, 128, self.zdim]
        #self.shapes = [self.data_dim, 128, 256, 512, 256, 128, self.zdim]
        self.shapes = [self.noise_dim+1, 256, 256, self.zdim]

        self.zeros = Zeros()
        self.ones = Ones()
        #self.initializer = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.5, seed=314159)
        self.initializer = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.02, seed=314159)

        self.is_built = False

    def __call__(self, noise, A, batch_size):
        #print(noise.shape)
        #print(A.shape)
        A = tf.dtypes.cast(A, tf.float32)
        noise = tf.dtypes.cast(noise, tf.float32)
        
        layer = tf.concat([noise, A], 1) #here we have Z dim = 114 [xdim (which includes adim) + ydim] + 1 from the real A == 115
        #batch_size = layer.shape[0]
        #print(layer.shape)
        
        if not self.is_built:
            self.Ws = [tf.Variable(self.zeros(shape=(self.shapes[i+1], self.shapes[i])), name='Gen_Ws') 
                                                                                    for i in range(len(self.shapes)-1)]
            self.bs = [tf.Variable(self.initializer(shape=(batch_size, self.shapes[i+1])), name='Gen_bs') 
                                                                                    for i in range(len(self.shapes)-1)]
            '''self.Ws = [tf.Variable(self.initializer(shape=(self.shapes[i+1], self.shapes[i])), name='Gen_Ws') 
                                                                                    for i in range(len(self.shapes)-1)]
            self.bs = [tf.Variable(self.zeros(shape=(batch_size, self.shapes[i+1])), name='Gen_bs') 
                                                                                    for i in range(len(self.shapes)-1)]'''

            self.is_built = True                                                                                

        prev_layer = layer
        
        for layer_idx in range(len(self.shapes[1:-1])):
            #print(layer_idx)
            #layer = tf.linalg.matmul(prev_layer, tf.transpose(self.Ws[layer_idx]))
            layer = tf.add(tf.linalg.matmul(prev_layer, tf.transpose(self.Ws[layer_idx])), self.bs[layer_idx])
            #layer = tf.nn.leaky_relu(layer)
            #layer = tf.nn.relu(layer)
            layer = tf.nn.sigmoid(layer)
            prev_layer = layer
        
        #layer = tf.linalg.matmul(prev_layer, tf.transpose(self.Ws[-1]))
        layer = tf.add(tf.linalg.matmul(prev_layer, tf.transpose(self.Ws[-1])), self.bs[-1]) #last layer

        #return tf.concat([tf.nn.relu(layer), A], 1)
        #return tf.concat([tf.nn.sigmoid(layer), A], 1)
        #return tf.concat([tf.nn.tanh(layer), A], 1)

        gen_data = tf.nn.sigmoid(layer)
        dec_data = self.dec(gen_data, batch_size)
        dec_data = tf.concat([dec_data, A], 1)
        return dec_data


######################################

class Discriminator2(tf.Module):
    def __init__(self, xdim, ydim, adim):
        super().__init__()
        '''self.zdim = xdim + ydim #in our dataset the protected attribute is included in the att vector x
        self.adim = adim'''
        self.data_dim = xdim + adim + ydim
        #self.hidden_layer = [256,128]#paper implementation
        #self.shapes = [self.data_dim, 256, 128, 1]
        self.shapes = [self.data_dim, 512, 256, 1]

        self.zeros = Zeros()
        self.ones = Ones()
        #self.initializer = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.5, seed=314159)
        self.initializer = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.02, seed=314159)

        self.is_built = False

    def __call__(self, data, batch_size):
        
        layer = data #must pass a tensor with x, a and y
        #batch_size = layer.shape[0]
        
        if not self.is_built:
            self.Ws = [tf.Variable(self.zeros(shape=(self.shapes[i+1], self.shapes[i])), name='Gen_Ws') 
                                                                                    for i in range(len(self.shapes)-1)]
            self.bs = [tf.Variable(self.ones(shape=(batch_size, self.shapes[i+1])), name='Gen_bs') 
                                                                                    for i in range(len(self.shapes)-1)]
            '''self.Ws = [tf.Variable(self.zeros(shape=(self.shapes[i+1], self.shapes[i])), name='D_Ws') 
                                                                                    for i in range(len(self.shapes)-1)]
            self.bs = [tf.Variable(self.ones(shape=(batch_size, self.shapes[i+1])), name='D_bs') 
                                                                                    for i in range(len(self.shapes)-1)]'''

            self.is_built = True                                                                                

        prev_layer = layer
        
        for layer_idx in range(len(self.shapes[1:-1])):

            layer = tf.add(tf.linalg.matmul(prev_layer, tf.transpose(self.Ws[layer_idx])), self.bs[layer_idx])
            #layer = tf.nn.leaky_relu(layer)
            layer = tf.nn.relu(layer)
            #layer = tf.nn.sigmoid(layer)
            prev_layer = layer
        
        layer = tf.add(tf.linalg.matmul(prev_layer, tf.transpose(self.Ws[-1])), self.bs[-1]) #last layer

        return tf.nn.sigmoid(layer)



######################################