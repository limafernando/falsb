import tensorflow as tf
from util import metrics

def train(model, X, Y, A, optimizer, faircoeff=1):
    
    '''training enc-clas-dec'''
   
    # enc_clas_dec = model.enc.variables + model.shl.variables + model.clas.variables
    # adv = model.enc.variables + model.shl.variables + model.adv.variables

    with tf.GradientTape() as tape_clas:
        
        # tape_clas.watch(enc_clas_dec)

        model(X, Y, A) #to compute the foward
        loss2min = model.loss #current loss
    
    grads = tape_clas.gradient(loss2min, model.variables)
    optimizer.apply_gradients(zip(grads, model.variables))

    '''training adv'''

    # with tf.GradientTape() as tape_adv:
    #     tape_adv.watch(adv)
    #     model(X, Y, A) #to compute the foward
    #     loss = -model.loss

    # grads_adv = tape_adv.gradient(loss, adv)
    # # grads_adv = -faircoeff * grads_adv
    # optimizer.apply_gradients(zip(grads_adv, adv))


def train_loop(model, raw_data, train_dataset, epochs, optmizer):
    
    print("> Epoch | Model Loss | Class Loss | Adv Loss | Class Acc | Adv Acc")

    #x_train, y_train, a_train = raw_data
    dataset_size = tf.data.experimental.cardinality(train_dataset).numpy()

    for epoch in range(epochs):
        Y_hat = None
        A_hat = None

        clas_acc = 0
        adv_acc = 0
        
        for X, Y, A in train_dataset:
            
            train(model, X, Y, A, optmizer)

            Y_hat = model.Y_hat
            A_hat = model.A_hat
            
            clas_acc += metrics.accuracy(Y, tf.math.round(Y_hat))
            adv_acc += metrics.accuracy(A, tf.math.round(A_hat))

        model_loss = tf.reduce_mean(model.loss)
        clas_loss = tf.reduce_mean(model.clas_loss)
        adv_loss = tf.reduce_mean(model.adv_loss)

        clas_acc = clas_acc / dataset_size
        adv_acc = adv_acc / dataset_size

        print("> {} | {} | {} | {} | {} | {}".format(
            epoch+1,
            model_loss, 
            clas_loss,
            adv_loss,
            clas_acc,
            adv_acc
        ))