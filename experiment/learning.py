from audioop import cross
from math import sqrt, isnan
from numpy import real

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from util import metrics

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def train(generator, discriminator, X, Y, A, batch_size, noise, gen_opt, disc_opt):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        X = tf.dtypes.cast(X, tf.float32)
        Y = tf.dtypes.cast(Y, tf.float32)
        A = tf.dtypes.cast(A, tf.float32)
        
        real_data = tf.concat((X, Y, A), 1)
        gen_data = generator(noise, A, batch_size)
        
        disc_real_data = discriminator(real_data, batch_size)
        disc_gen_data = discriminator(gen_data, batch_size)
        
        gen_loss = generator_loss(disc_gen_data)
        disc_loss = discriminator_loss(disc_real_data, disc_gen_data)
    
    gen_grads = gen_tape.gradient(gen_loss, generator.variables)
    disc_grads = disc_tape.gradient(disc_loss, discriminator.variables)
        
    gen_opt.apply_gradients(zip(gen_grads, generator.variables))
    disc_opt.apply_gradients(zip(disc_grads,discriminator.variables))

    return gen_loss, disc_loss

def train_loop(generator, discriminator, train_dataset, batch_size, noise_dim, epochs, gen_opt=None, disc_opt=None, d2_opt=None):
    """Loop function for the FairGAN training"""
    
    print("> Epoch | G Loss | Disc Loss")

    for epoch in range(epochs):
        epoch_gen_loss = None
        epoch_disc_loss = None
        #batch_count = 1
        
        batch_gen_loss = []
        batch_disc_loss = []

        for X, Y, A in train_dataset:            
            noise = tf.random.normal([batch_size, noise_dim])
            #print('shape noise: {}'.format(noise.shape))
            gen_loss, disc_loss = train(generator, discriminator, X, Y, A, batch_size, noise, gen_opt, disc_opt)
            
            batch_gen_loss.append(gen_loss)
            batch_disc_loss.append(disc_loss)
        
        epoch_gen_loss = tf.reduce_mean(batch_gen_loss)
        epoch_disc_loss = tf.reduce_mean(batch_disc_loss)

        print("> {} | {} | {}".format(
            epoch+1,
            epoch_gen_loss,
            epoch_disc_loss))