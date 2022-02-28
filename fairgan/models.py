import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.initializers import GlorotNormal, Zeros, Ones
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.python.keras.backend import reset_uids
from tensorflow.python.ops.gen_batch_ops import batch

class Encoder(tf.Module):
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

class Decoder(tf.Module):
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

class Autoencoder(tf.Module):
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

        self.enc = Encoder(self.xdim, self.ydim, self.adim)
        self.dec = Decoder(self.xdim, self.ydim, self.adim)

    def __call__(self, X, Y, A):
        #ensure casting
        self.X = tf.dtypes.cast(X, tf.float32)
        self.Y = tf.dtypes.cast(Y, tf.float32)
        self.A = tf.dtypes.cast(A, tf.float32)

        #self.real_data = tf.concat([self.X, self.Y, self.A], 1) #in our dataset the protected attribute is included in the att vector x
        self.real_data = tf.concat([self.X, self.Y], 1)
        self.enc_repr = self.enc(self.real_data, self.batch_size)
        self.dec_repr = self.dec(self.enc_repr, self.batch_size)

        self.ae_loss = self.get_ae_loss(self.real_data, self.dec_repr)

    def get_ae_loss(self, real_data, dec_repr):
        '''it is the squared euclidean norm'''
        
        '''loss = tf.sqrt(
                tf.math.reduce_sum(
                    tf.math.squared_difference(real_data, dec_repr)
                )
        )''' #this old implementation don't consider the square of the euclidean norm

        #both bellow implementations give the same result

        loss = tf.square(tf.norm(tensor=(real_data - dec_repr)))
        '''loss = tf.math.reduce_sum(
                    tf.math.squared_difference(real_data, dec_repr)
                )'''

        return loss

######################################

class Generator(tf.Module):
    #takes as input the protected attribute (a) and the noise variable (z)

    def __init__(self, xdim, ydim, adim):
        super().__init__()
        '''self.zdim = xdim + ydim #in our dataset the protected attribute is included in the att vector x
        self.adim = adim'''
        self.zdim = xdim + ydim #random noise dimension
        self.data_dim = xdim + adim + ydim
        self.hidden_layer = [128,128] #paper implementation
        #self.shapes = [self.data_dim, 128, 256, 512, 512, 256, 128, self.zdim]
        self.shapes = [self.data_dim, 256, 256, self.zdim]

        self.zeros = Zeros()
        self.ones = Ones()
        self.initializer = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.5, seed=314159)

        self.is_built = False

    def __call__(self, noise, A, batch_size):
        #print(noise.shape)
        #print(A.shape)
        layer = tf.concat([noise, A], 1) #here we have Z dim = 114 [xdim (which includes adim) + ydim] + 1 from the real A == 115
        #batch_size = layer.shape[0]
        #print(layer.shape)
        
        if not self.is_built:
            self.Ws = [tf.Variable(self.zeros(shape=(self.shapes[i+1], self.shapes[i])), name='Gen_Ws') 
                                                                                    for i in range(len(self.shapes)-1)]
            self.bs = [tf.Variable(self.initializer(shape=(batch_size, self.shapes[i+1])), name='Gen_bs') 
                                                                                    for i in range(len(self.shapes)-1)]

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

        #return tf.nn.relu(layer)
        return tf.nn.sigmoid(layer)
    

class GDec(tf.Module):
    def __init__(self, generator, decoder):
        super().__init__() #GDec = Dec(G(z))
        self.gen = generator
        self.dec = decoder

    def __call__(self, noise, A, batch_size):
        gen_data = self.gen(noise, A, batch_size)
        
        dec_data = self.dec(gen_data, batch_size)
        dec_data = tf.concat([dec_data, A], 1)
        #dec_data = tf.concat([gen_data, A], 1)
        return dec_data, gen_data

    '''def get_gdec_loss(self, real_data, dec_repr):
        #euclidian distance
        #dec_repr - tf.concat(X, Y, A)
        loss = tf.sqrt(
                tf.math.reduce_sum(
                    tf.math.squared_difference(real_data, dec_repr)
                )
        )

        return loss'''


######################################

class Discriminator(tf.Module):
    def __init__(self, xdim, ydim, adim):
        super().__init__()
        '''self.zdim = xdim + ydim #in our dataset the protected attribute is included in the att vector x
        self.adim = adim'''
        self.data_dim = xdim + adim + ydim
        #self.hidden_layer = [256,128]paper implementation
        self.shapes = [self.data_dim, 512, 256, 1]

        self.zeros = Zeros()
        self.ones = Ones()

        self.is_built = False

    def __call__(self, data, batch_size):
        
        layer = data #must pass a tensor with x, a and y
        #batch_size = layer.shape[0]
        
        if not self.is_built:
            self.Ws = [tf.Variable(self.zeros(shape=(self.shapes[i+1], self.shapes[i])), name='D_Ws') 
                                                                                    for i in range(len(self.shapes)-1)]
            self.bs = [tf.Variable(self.ones(shape=(batch_size, self.shapes[i+1])), name='D_bs') 
                                                                                    for i in range(len(self.shapes)-1)]

            self.is_built = True                                                                                

        prev_layer = layer
        
        for layer_idx in range(len(self.shapes[1:-1])):

            layer = tf.add(tf.linalg.matmul(prev_layer, tf.transpose(self.Ws[layer_idx])), self.bs[layer_idx])
            #layer = tf.nn.leaky_relu(layer)
            layer = tf.nn.relu(layer)
            prev_layer = layer
        
        layer = tf.add(tf.linalg.matmul(prev_layer, tf.transpose(self.Ws[-1])), self.bs[-1]) #last layer

        return tf.nn.sigmoid(layer)

class FairDiscriminator(tf.Module):
    def __init__(self, xdim, ydim, adim):
        super().__init__()
        '''self.zdim = xdim + ydim #in our dataset the protected attribute is included in the att vector x
        self.adim = adim'''
        self.data_dim = xdim + ydim #here the d2 receive an input of x and y to predict a
        self.adim = adim
        #self.hidden_layer = [256,128]paper implementation
        self.shapes = [self.data_dim, 512, 256, 1]

        self.zeros = Zeros()
        self.ones = Ones()

        self.is_built = False

    def __call__(self, data, batch_size):
        
        layer = data #must pass only a tensor with x and y
        #batch_size = layer.shape[0]
        
        if not self.is_built:
            self.Ws = [tf.Variable(self.zeros(shape=(self.shapes[i+1], self.shapes[i])), name='D_Ws') 
                                                                                for i in range(len(self.shapes)-1)]
            self.bs = [tf.Variable(self.ones(shape=(batch_size, self.shapes[i+1])), name='D_bs') 
                                                                                for i in range(len(self.shapes)-1)]

            self.is_built = True                                                                                

        prev_layer = layer
        
        for layer_idx in range(len(self.shapes[1:-1])):

            layer = tf.add(tf.linalg.matmul(prev_layer, tf.transpose(self.Ws[layer_idx])), self.bs[layer_idx])
            #layer = tf.nn.leaky_relu(layer)
            layer = tf.nn.relu(layer)
            prev_layer = layer
        
        layer = tf.add(tf.linalg.matmul(prev_layer, tf.transpose(self.Ws[-1])), self.bs[-1]) #last layer

        return tf.nn.sigmoid(layer)

######################################

class FairGAN(tf.Module):
    def __init__(self, xdim, ydim, adim, decoder, batch_size):
        super().__init__()
        
        self.xdim = xdim #input dimensions
        self.ydim = ydim #label dimension
        self.adim = adim #sensitive atribute dimension
        self.batch_size = batch_size
        self.v1_loss, self.v2_loss = None, None

        self.gen = Generator(self.xdim, self.ydim, self.adim)
        self.gdec = GDec(self.gen, decoder)
        self.d1 = Discriminator(self.xdim, self.ydim, self.adim)
        self.d2 = FairDiscriminator(self.xdim, self.ydim, self.adim)

    def __call__(self, X, Y, A, noise, step):
        #ensure casting
        self.X = tf.dtypes.cast(X, tf.float32)
        self.Y = tf.dtypes.cast(Y, tf.float32)
        self.A = tf.dtypes.cast(A, tf.float32)
        self.noise = tf.dtypes.cast(noise, tf.float32)
        #print(self.Z.shape)

        if step == "v1":
            #gen_data = self.gen(self.Z, self.A, self.batch_size)
            #gen_data = self.gdec(tf.concat([gen_data, self.A], 1), self.batch_size)

            gen_data, b = self.gdec(self.noise, self.A, self.batch_size)

            real_data = tf.concat((self.X, self.Y, self.A), 1)
            
            d1_real_data = self.d1(real_data, self.batch_size)
            d1_gen_data = self.d1(gen_data, self.batch_size)
            
            '''real_loss = self.get_v1_loss(tf.ones_like(d1_real_data), d1_real_data)
            gen_loss = self.get_v1_loss(tf.zeros_like(d1_gen_data), d1_gen_data)'''
            
            #self.v1_loss = real_loss + gen_loss
            self.v1_loss = self.get_v1_loss(d1_real_data, d1_gen_data)
            self.gdec_loss = self.get_gdec_loss(tf.ones_like(d1_gen_data), d1_gen_data)

        elif step == "v2":
            As2male = tf.zeros_like(self.A)
            As2female = tf.ones_like(self.A)

            #gen_male_data = self.gen(self.Z, As2male, self.batch_size)
            #gen_male_data = self.gdec(tf.concat([gen_male_data, As2male], 1), self.batch_size)
            gen_male_data, b = self.gdec(self.noise, As2male, self.batch_size)
            gen_male_data = gen_male_data[:,:-1]

            #gen_female_data = self.gen(self.Z, As2female, self.batch_size)
            #gen_female_data = self.gdec(tf.concat([gen_female_data, As2female],1), self.batch_size)
            gen_female_data, b = self.gdec(self.noise, As2female, self.batch_size)
            gen_female_data = gen_female_data[:,:-1]

            d2_male_data = self.d2(gen_male_data, self.batch_size)
            d2_female_data = self.d2(gen_female_data, self.batch_size)

            '''male_loss = self.get_v2_loss(As2male, d2_male_data) #male data is 0
            female_loss = self.get_v2_loss(As2female, d2_female_data) #female data is 1'''
            
            #self.v2_loss = male_loss + female_loss
            self.v2_loss = self.get_v2_loss(d2_male_data, d2_female_data)

        else: 
            print("invalid step for FairGAN")


    '''def get_v1_loss(self, real_output, d1_output):
        bce = BinaryCrossentropy(from_logits=False)
        return bce(real_output, d1_output)
        #loss ='''

    def get_v1_loss(self, d1_real_data, d1_gen_data):
        real_loss = tf.math.log(d1_real_data)
        gen_loss = tf.math.log(1-d1_gen_data)
        v1_loss = real_loss + gen_loss
        return v1_loss        

    '''def get_v2_loss(self, real_output, d2_output):
        bce = BinaryCrossentropy(from_logits=False)
        return bce(real_output, d2_output)'''

    def get_v2_loss(self, d2_male_data, d2_female_data):
        female_loss = tf.math.log(d2_female_data)
        male_loss = tf.math.log(1-d2_male_data)
        v1_loss = female_loss+male_loss
        return v1_loss    

    def get_gdec_loss(self, ones_like_output, disc_output):
        #measure how well the generator is fooling the discriminator
        bce = BinaryCrossentropy(from_logits=False)
        return bce(ones_like_output, disc_output)

    def get_gen_loss(self, real_data, dec_repr):
        '''euclidian distance'''
        #dec_repr - tf.concat(X, Y, A)
        loss = tf.sqrt(
                tf.math.reduce_sum(
                    tf.math.squared_difference(real_data, dec_repr)
                )
        )

        return loss