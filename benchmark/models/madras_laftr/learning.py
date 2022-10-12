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

    #x_train, y_train, a_train = raw_data
    dataset_size = tf.data.experimental.cardinality(train_dataset).numpy()

    for epoch in range(epochs):
        Y_hat = None
        A_hat = None
        X_hat = None

        clas_acc = 0
        adv_acc = 0
        dec_acc = 0
        
        for X, Y, A in train_dataset:
            
            train(model, X, Y, A, optmizer)

            Y_hat = model.Y_hat
            A_hat = model.A_hat
            X_hat = model.X_hat
            clas_acc += metrics.accuracy(Y, tf.math.round(Y_hat))
            adv_acc += metrics.accuracy(A, tf.math.round(A_hat))
            dec_acc += metrics.accuracy(X, X_hat)

        model_loss = tf.reduce_mean(model.loss)
        clas_loss = tf.reduce_mean(model.clas_loss)
        adv_loss = tf.reduce_mean(model.adv_loss)
        dec_loss = tf.reduce_mean(model.recon_loss)

        clas_acc = clas_acc / dataset_size
        adv_acc = adv_acc / dataset_size
        dec_acc = dec_acc / dataset_size

        print("> {} | {} | {} | {} | {} | {} | {} | {}".format(
            epoch+1,
            model_loss, 
            clas_loss,
            adv_loss,
            dec_loss, 
            clas_acc,
            adv_acc,
            dec_acc))