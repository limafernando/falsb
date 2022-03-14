import tensorflow as tf

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