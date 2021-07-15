import tensorflow as tf
from util import metrics

def train(model, X, Y, A, optimizer):
    
    '''training enc-clas-dec'''
   
    enc_clas_dec = model.enc.variables + model.clas.variables + model.dec.variables
    adv = model.adv.variables

    with tf.GradientTape() as tape_min:
        
        tape_min.watch(enc_clas_dec)

        model(X, Y, A) #to compute the foward
        loss2min = model.loss #current loss
    
    grads = tape_min.gradient(loss2min, enc_clas_dec)
    optimizer.apply_gradients(zip(grads, enc_clas_dec))

    '''training adv'''

    with tf.GradientTape() as tape_max:
        tape_max.watch(adv)
        model(X, Y, A) #to compute the foward
        loss2max = -model.loss

    grads_adv = tape_max.gradient(loss2max, adv)
    optimizer.apply_gradients(zip(grads_adv, adv))


def train_loop(model, raw_data, train_dataset, epochs, optmizer):
    
    print("> Epoch | Model Loss | Class Loss | Adv Loss | Dec Loss | Class Acc | Adv Acc | Dec Acc")

    x_train, y_train, a_train = raw_data

    for epoch in range(epochs):
        Y_hat = None
        A_hat = None
        X_hat = None
        batch_count = 1
        
        for X, Y, A in train_dataset:
            
            train(model, X, Y, A, optmizer)

            if batch_count == 1:
                Y_hat = model.Y_hat
                A_hat = model.A_hat
                X_hat = model.X_hat
                batch_count += 1
                
            else:
                Y_hat = tf.concat([Y_hat, model.Y_hat], 0)
                A_hat = tf.concat([A_hat, model.A_hat], 0)
                X_hat = tf.concat([X_hat, model.X_hat], 0)

        model_loss = tf.reduce_mean(model.loss)
        clas_loss = tf.reduce_mean(model.clas_loss)
        adv_loss = tf.reduce_mean(model.adv_loss)
        dec_loss = tf.reduce_mean(model.recon_loss)
        clas_acc = metrics.accuracy(y_train, tf.math.round(Y_hat))
        adv_acc = metrics.accuracy(a_train, tf.math.round(A_hat))
        dec_acc = metrics.accuracy(x_train, X_hat)

        print("> {} | {} | {} | {} | {} | {} | {} | {}".format(
            epoch+1,
            model_loss, 
            clas_loss,
            adv_loss,
            dec_loss, 
            clas_acc,
            adv_acc,
            dec_acc))