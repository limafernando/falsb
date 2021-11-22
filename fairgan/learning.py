from math import sqrt, isnan

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from util import metrics

from fairgan.models import *

def pre_train(model, X, Y, A, optimizer):
    """Function responsible for the Autoencoder training"""
    
    ae_vars = model.enc.variables + model.dec.variables

    with tf.GradientTape() as tape:
        
        tape.watch(ae_vars)

        model(X, Y, A) #to compute the foward
        ae_loss = model.ae_loss #current loss
    
    grads = tape.gradient(ae_loss, ae_vars)
    optimizer.apply_gradients(zip(grads, ae_vars))

def pre_train_loop(model, raw_data, train_dataset, epochs, opt=None):
    """Loop function for the Autoencoder training"""
    
    print("> Epoch | Model Loss")

    x_train, y_train, a_train = raw_data

    for epoch in range(epochs):
        dec_repr = None
        batch_count = 1
        
        for X, Y, A in train_dataset:
            
            pre_train(model, X, Y, A, opt)

            if batch_count == 1:
                dec_repr = model.dec_repr
                batch_count += 1
                
            else:
                dec_repr = tf.concat([dec_repr, model.dec_repr], 0)

        ae_loss = tf.reduce_mean(model.ae_loss)

        print("> {} | {}".format(
            epoch+1,
            ae_loss))

#############################################################################

def train(model, X, Y, A, Z, d1_opt, d2_opt):
    """Function responsible for the FairGAN training"""
    
    d1_vars = model.gdec.variables + model.d1.variables
    d2_vars = model.gdec.variables + model.d2.variables

    with tf.GradientTape() as v1_tape:
        
        v1_tape.watch(d1_vars)

        model(X, Y, A, Z, 'v1') #to compute the foward
        v1_loss = model.v1_loss #v1 loss
    
    grads = v1_tape.gradient(v1_loss, d1_vars)
    d1_opt.apply_gradients(zip(grads, d1_vars))

    with tf.GradientTape() as v2_tape:
        
        v2_tape.watch(d2_vars)

        model(X, Y, A, Z, 'v2') #to compute the foward
        v2_loss = model.v2_loss #v2 loss

    grads = v2_tape.gradient(v2_loss, d2_vars)
    d2_opt.apply_gradients(zip(grads, d2_vars))

def train_loop(model, raw_data, train_dataset, batch_size, noise_dim, epochs, d1_opt=None, d2_opt=None):
    """Loop function for the FairGAN training"""
    
    print("> Epoch | D1 Loss | D2 Loss")

    for epoch in range(epochs):
        
        #batch_count = 1
        
        for X, Y, A in train_dataset:
            noise = tf.random.normal([batch_size, noise_dim])
            #print('shape noise: {}'.format(noise.shape))
            train(model, X, Y, A, noise, d1_opt, d2_opt)

        v1_loss = tf.reduce_mean(model.v1_loss)
        v2_loss = tf.reduce_mean(model.v2_loss)

        print("> {} | {} | {}".format(
            epoch+1,
            v1_loss,
            v2_loss))