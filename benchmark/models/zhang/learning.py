from math import isnan


import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


from util import metrics


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


def train_loop(model, raw_data, train_dataset, epochs, opt=None):
    
    print("> Epoch | Class Loss | Adv Loss | Class Acc | Adv Acc")

    #x_train, y_train, a_train = raw_data
    dataset_size = tf.data.experimental.cardinality(train_dataset).numpy()

    if opt is not None:
        optimizer = opt
        decay4epoch = False
    else:
        decay4epoch = True
    
    for epoch in range(epochs):
        Y_hat = None
        A_hat = None
        
        clas_acc = 0
        adv_acc = 0
        
        alpha=1/(epoch+1)#sqrt(epoch+1)

        if decay4epoch:
            lr = 0.001/(epoch+1)
            optimizer = Adam(learning_rate=lr)
        
        for X, Y, A in train_dataset:
            
            r = train(model, X, Y, A, optimizer, alpha)

            if r:
                print('parou')
                print(model.clas_loss, model.adv_loss)
                break

            Y_hat = model.Y_hat
            A_hat = model.A_hat
            clas_acc += metrics.accuracy(Y, tf.math.round(Y_hat))
            adv_acc += metrics.accuracy(A, tf.math.round(A_hat))

        clas_loss = model.clas_loss
        adv_loss = model.adv_loss
        clas_acc = clas_acc / dataset_size
        adv_acc = adv_acc / dataset_size
    
        print("> {} | {} | {} | {} | {}".format(
            epoch+1, 
            clas_loss,
            adv_loss, 
            clas_acc,
            adv_acc))
