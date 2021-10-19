from math import sqrt, isnan

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from util import metrics

from fairgan.models import *

def pre_train(model, X, Y, A, optimizer, alpha=1):
    """Function responsible for the Autoencoder training"""
    
    aue_vars = model.enc.variables + model.dec.variables

    with tf.GradientTape() as tape:
        
        tape.watch(aue_vars)

        model(X, Y, A) #to compute the foward
        loss = model.loss #current loss
    
    grads = tape.gradient(loss, aue_vars)
    optimizer.apply_gradients(zip(grads, aue_vars))

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

        model_loss = tf.reduce_mean(model.loss)

        print("> {} | {}".format(
            epoch+1,
            model_loss))

#############################################################################

def train(model, X, Y, A, optimizer, alpha=1):
    """Function responsible for the FairGAN training"""
    pass

def train_loop(model, raw_data, train_dataset, epochs, opt=None):
    """Loop function for the FairGAN training"""
    
    print("> Epoch | Class Loss | Adv Loss | Class Acc | Adv Acc")

    pass