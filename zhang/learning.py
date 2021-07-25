from math import sqrt, isnan

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from util import metrics

from zhang.models import FairLogisticRegression

def projection(V, A):
    if not np.all(V.numpy()):
        shape = [1, A.shape[1]]
        zeros = tf.zeros(shape, dtype='float32')
        return zeros

    else:
        P = tf.multiply(V, A)
        P = tf.multiply(P, V)
        P = tf.divide(P, tf.norm(V))
        return P

def train(model, X, Y, A, optimizer, alpha=1):
    adv_vars = [model.adv.U, model.b, model.adv.c]
    clas_vars = [model.clas.W, model.b]
    
    with tf.GradientTape() as adv_tape, tf.GradientTape(persistent=True) as clas_tape:
        
        model(X, Y, A) #to compute the foward
        adv_loss = model.adv_loss #current adversarial loss
        clas_loss = model.clas_loss #current classifier loss
        model_loss = model.model_loss

    if isnan(adv_loss) or isnan(clas_loss):
        print('any loss is NaN')
        return True
    
    dULa = adv_tape.gradient(adv_loss, adv_vars) #adv_grads
    optimizer.apply_gradients(zip(dULa, adv_vars))

    dWLp = clas_tape.gradient(clas_loss, clas_vars) #regular grads for classifier

    dWLa = clas_tape.gradient(adv_loss, clas_vars) #grads for W with the adversarial loss
    
    proj_dWLa_dWLp = [] #prevents the classifier from moving in a direction that helps the adversary decrease its loss
    for i in range(len(dWLa)):
        proj_dWLa_dWLp.append(projection(dWLa[i], dWLp[i]))
   
    max_adv_loss = [] #terms that attemps to increase adv loss
    for i in range(len(dWLa)):
        max_adv_loss.append(tf.math.multiply(alpha, dWLa[i]))
    
    proj_minus_max_adv_loss = [] 
    for i in range(len(max_adv_loss)):
        proj_minus_max_adv_loss.append(tf.subtract(proj_dWLa_dWLp[i], max_adv_loss[i])) 

    clas_grads = []
    for i in range(len(dWLa)):
        clas_grads.append(tf.subtract(dWLp[i],  proj_minus_max_adv_loss[i]))

    optimizer.apply_gradients(zip(clas_grads, clas_vars))
    
    model(X, Y, A) #to compute the foward
    return False

def train_loop(model, raw_data, train_dataset, epochs, optimizer):
    
    print("> Epoch | Class Loss | Adv Loss | Class Acc | Adv Acc")

    x_train, y_train, a_train = raw_data
    
    for epoch in range(epochs):
        Y_hat = None
        A_hat = None
        batch_count = 1
        
        alpha=1/sqrt(epoch+1)
        
        for X, Y, A in train_dataset:
            
            r = train(model, X, Y, A, optimizer, alpha)

            if r:
                print('parou')
                print(model.clas_loss, model.adv_loss)
                break

            if batch_count == 1:
                Y_hat = model.Y_hat
                A_hat = model.A_hat
                batch_count += 1
            else:
                Y_hat = tf.concat([Y_hat, model.Y_hat], 0)
                A_hat = tf.concat([A_hat, model.A_hat], 0)

        clas_loss = model.clas_loss
        adv_loss = model.adv_loss
        clas_acc = metrics.accuracy(y_train, tf.math.round(Y_hat))
        adv_acc = metrics.accuracy(a_train, tf.math.round(A_hat))
    
        print("> {} | {} | {} | {} | {}".format(
            epoch+1, 
            clas_loss,
            adv_loss, 
            clas_acc,
            adv_acc))