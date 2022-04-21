import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from util import metrics


def train(model, X, Y, A, optimizer):
    clas_vars = [model.clas.W, model.b]
    
    with tf.GradientTape() as clas_tape:
        
        model(X, Y, A)        
        clas_loss = model.clas_loss

    dWLp = clas_tape.gradient(clas_loss, clas_vars)
    optimizer.apply_gradients(zip(dWLp, clas_vars))


def train_loop(model, train_dataset, epochs, opt=None):
    dataset_size = tf.data.experimental.cardinality(train_dataset).numpy()
    print("> Epoch | Class Loss | Class Acc")

    if opt is not None:
        optimizer = opt
        decay4epoch = False
    else:
        decay4epoch = True
    
    for epoch in range(epochs):
        Y_hat = None
        clas_acc = 0

        if decay4epoch:
            lr = 0.001/(epoch+1)
            optimizer = Adam(learning_rate=lr)
        
        for X, Y, A in train_dataset:
            
            r = train(model, X, Y, A, optimizer)
            if r:
                print('parou')
                print(model.clas_loss)
                break

            Y_hat = model.Y_hat

            clas_acc += metrics.accuracy(Y, tf.math.round(Y_hat))

        clas_loss = model.clas_loss
        clas_acc = clas_acc / dataset_size
    
        print("> {} | {} | {}".format(
            epoch+1, 
            clas_loss, 
            clas_acc))