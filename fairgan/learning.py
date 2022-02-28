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
def v1(model, X, Y, A, noise, gen_opt, d1_opt):
    '''with tf.GradientTape() as gen_tape, tf.GradientTape() as v1_tape:
    #with tf.GradientTape(persistent=True) as tape:
            
        #gen_tape.watch(model.gdec.variables)
        #v1_tape.watch(model.d1.variables)

        model(X, Y, A, noise, 'v1') #to compute the foward
        v1_loss = model.v1_loss #v1 loss
        #gdec_loss = model.gdec_loss
        loss2max = -v1_loss
    
    d1_grads = v1_tape.gradient(loss2max, model.d1.variables)
    d1_opt.apply_gradients(zip(d1_grads, model.d1.variables))

    gen_grads = gen_tape.gradient(v1_loss, model.gdec.variables)
    d1_opt.apply_gradients(zip(gen_grads, model.gdec.variables))'''

    with tf.GradientTape() as v1_tape:
            
        #gen_tape.watch(model.gdec.variables)
        v1_tape.watch(model.d1.variables)

        model(X, Y, A, noise, 'v1') #to compute the foward
        #v1_loss = model.v1_loss #v1 loss
        #gdec_loss = model.gdec_loss
        loss2max = -model.v1_loss
    
    d1_grads = v1_tape.gradient(loss2max, model.d1.variables)
    d1_opt.apply_gradients(zip(d1_grads, model.d1.variables))
    
    with tf.GradientTape() as gen_tape:
            
        gen_tape.watch(model.gdec.variables)
        #v1_tape.watch(model.d1.variables)

        model(X, Y, A, noise, 'v1') #to compute the foward
        v1_loss = model.v1_loss #v1 loss
        #gdec_loss = model.gdec_loss
        #loss2max = -v1_loss

    gen_grads = gen_tape.gradient(v1_loss, model.gdec.variables)
    d1_opt.apply_gradients(zip(gen_grads, model.gdec.variables))
    

    
        
def v2(model, X, Y, A, noise, gen_opt, d1_opt, d2_opt):
    v1(model, X, Y, A, noise, gen_opt, d1_opt)
    
    '''#with tf.GradientTape() as gen_tape, tf.GradientTape() as v2_tape:
    with tf.GradientTape(persistent=True) as tape:
        
        #gen_tape.watch(model.gdec.variables)
        #v2_tape.watch(model.d2.variables)

        model(X, Y, A, noise, 'v2') #to compute the foward
        v2_loss = model.v2_loss #v2 loss

    grads = tape.gradient(v2_loss, model.gdec.variables)
    gen_opt.apply_gradients(zip(grads, model.gdec.variables))
        
    grads = tape.gradient(v2_loss, model.d2.variables)
    d2_opt.apply_gradients(zip(grads, model.d2.variables))'''

    with tf.GradientTape() as v2_tape:
            
        #gen_tape.watch(model.gdec.variables)
        v2_tape.watch(model.d2.variables)

        model(X, Y, A, noise, 'v2') #to compute the foward
        #v1_loss = model.v1_loss #v1 loss
        #gdec_loss = model.gdec_loss
        loss2max = -model.v2_loss
    
    d2_grads = v2_tape.gradient(loss2max, model.d2.variables)
    d2_opt.apply_gradients(zip(d2_grads, model.d2.variables))
    
    with tf.GradientTape() as gen_tape:
            
        gen_tape.watch(model.gdec.variables)
        #v1_tape.watch(model.d1.variables)

        model(X, Y, A, noise, 'v2') #to compute the foward
        v2_loss = model.v2_loss #v1 loss
        #gdec_loss = model.gdec_loss
        #loss2max = -v1_loss

    gen_grads = gen_tape.gradient(v2_loss, model.gdec.variables)
    d2_opt.apply_gradients(zip(gen_grads, model.gdec.variables))


def train(model, X, Y, A, Z, phase, gen_opt, d1_opt, d2_opt):
    """Function responsible for the FairGAN training"""
    
    #d1_vars = model.gdec.variables + model.d1.variables
    #d2_vars = model.gdec.variables + model.d2.variables
    if phase == 'v1':
        v1(model, X, Y, A, Z, gen_opt, d1_opt)        

    elif phase =='v2':
        v2(model, X, Y, A, Z, gen_opt, d1_opt, d2_opt)
        
    else:
        pass

def train_loop(model, raw_data, train_dataset, batch_size, noise_dim, epochs, phase, gen_opt=None, d1_opt=None, d2_opt=None):
    """Loop function for the FairGAN training"""
    
    print("> Epoch | Phase | D1 Loss | D2 Loss | Gdec Loss")

    for epoch in range(epochs):
        
        #batch_count = 1
        
        for X, Y, A in train_dataset:
            noise = tf.random.normal([batch_size, noise_dim])
            #print('shape noise: {}'.format(noise.shape))
            train(model, X, Y, A, noise, phase, gen_opt, d1_opt, d2_opt)

        if phase == 'v1':
            v1_loss = tf.reduce_mean(model.v1_loss)
            v2_loss = '---'
            gen_loss = tf.reduce_mean(model.gdec_loss)
        elif phase == 'v2':
            v1_loss = tf.reduce_mean(model.v1_loss)
            v2_loss = tf.reduce_mean(model.v2_loss)
            gen_loss = tf.reduce_mean(model.gdec_loss)
        else:
            pass

        print("> {} | {} | {} | {} | {}".format(
            epoch+1,
            phase,
            v1_loss,
            v2_loss,
            gen_loss))