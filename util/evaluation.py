import tensorflow as tf
import numpy as np

from util import metrics

def evaluation(model, valid_data):
    Y_hat = None
    A_hat = None
    batch_count = 1
    
    for X, Y, A in valid_data:
        
        model(X, Y, A)
        
        if batch_count == 1:
            Y_hat = model.Y_hat
            A_hat = model.A_hat
            batch_count += 1
        else:
            Y_hat = tf.concat([Y_hat, model.Y_hat], 0)
            A_hat = tf.concat([A_hat, model.A_hat], 0)
    
    return Y_hat, A_hat

def compute_metrics(Y, Y_hat, A, A_hat):
    Y_hat = tf.math.round(Y_hat)
    A_hat = tf.math.round(A_hat)
    
    clas_acc = metrics.accuracy(Y, Y_hat)
    adv_acc = metrics.accuracy(A, A_hat)

    print("> Class Acc | Adv Acc")
    print("> {} | {}".format(clas_acc, adv_acc))

    dp = metrics.DP(Y_hat.numpy(), A)
    deqodds = metrics.DEqOdds(Y, Y_hat.numpy(), A)
    deqopp = metrics.DEqOpp(Y, Y_hat.numpy(), A)

    print("> DP | DEqOdds | DEqOpp")
    print("> {} | {} | {}".format(dp, deqodds, deqopp))

    tp = metrics.TP(Y, Y_hat.numpy())
    tn = metrics.TN(Y, Y_hat.numpy())
    fp = metrics.FP(Y, Y_hat.numpy())
    fn = metrics.FN(Y, Y_hat.numpy())

    print('> Confusion Matrix \n' +
                'TN: {} | FP: {} \n'.format(tn, fp) +
                'FN: {} | TP: {}'.format(fn, tp))

    m = [metrics.TN, metrics.FP, metrics.FN, metrics.TP]
    metrics_a0 = [0, 0, 0, 0]
    metrics_a1 = [0, 0, 0, 0]
    for i in range(len(m)):
        metrics_a0[i] = metrics.subgroup(m[i], A, Y, Y_hat.numpy())
        metrics_a1[i] = metrics.subgroup(m[i], 1 - A, Y, Y_hat.numpy())

    print('> Confusion Matrix for A = 0 \n' +
            'TN: {} | FP: {} \n'.format(metrics_a0[0], metrics_a0[1]) +
            'FN: {} | TP: {}'.format(metrics_a0[2], metrics_a0[3]))

    print('> Confusion Matrix for A = 1 \n' +
            'TN: {} | FP: {} \n'.format(metrics_a1[0], metrics_a1[1]) +
            'FN: {} | TP: {}'.format(metrics_a1[2], metrics_a1[3]))

    confusion_matrix = np.array([[tn, fp],
                                [fn, tp]])

    return clas_acc, dp, deqodds, deqopp, confusion_matrix, metrics_a0, metrics_a1

def compute_tradeoff(performance_metric, fairness_metric):
    tradeoff = 2*(performance_metric*fairness_metric)/(performance_metric+fairness_metric)
    return tradeoff