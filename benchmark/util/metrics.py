import numpy as np
import tensorflow as tf

eps = 1e-12

def main():
    Y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    Ypred = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1, 0.2, 0.3, 0.8, 0.9])
    A = np.array([1, 1, 0, 1, 0, 0, 0, 1, 0, 1])

    assert pos(Y) == 5
    assert neg(Y) == 5
    assert pos(Ypred) == 5
    assert neg(Ypred) == 5

    assert TP(Y, Ypred) == 3
    assert FP(Y, Ypred) == 2
    assert TN(Y, Ypred) == 3
    assert FN(Y, Ypred) == 2

    assert np.isclose(TPR(Y, Ypred), 0.6)
    assert np.isclose(FPR(Y, Ypred) , 0.4)
    assert np.isclose(TNR(Y, Ypred) , 0.6)
    assert np.isclose(FNR(Y, Ypred) , 0.4)
    assert np.isclose(calibPosRate(Y, Ypred) , 0.6)
    assert np.isclose(calibNegRate(Y, Ypred) , 0.6)
    assert np.isclose(errRate(Y, Ypred) , 0.4)
    assert np.isclose(accuracy(Y, Ypred) , 0.6)
    assert np.isclose(subgroup(TNR, A, Y, Ypred) , 0.5)
    assert np.isclose(subgroup(pos, 1 - A, Ypred) , 2)
    assert np.isclose(subgroup(neg, 1 - A, Y) , 3)
    assert np.isclose(DI_FP(Y, Ypred, A) , abs(1.0 / 6))
    assert np.isclose(DI_FP(Y, Ypred, 1 - A) , abs(1.0 / 6))
    assert np.isclose(DI_FN(Y, Ypred, A) , abs(1.0 / 6))
    assert np.isclose(DI_FN(Y, Ypred, 1 - A) , abs(1.0 / 6))
    assert np.isclose(subgroup(accuracy, A, Y, Ypred), 0.6)
    assert np.isclose(subgroup(errRate, 1 - A, Y, Ypred), 0.4)

def pos(Y):
    return np.sum(np.round(Y)).astype(np.float32)

def neg(Y):
    return np.sum(np.logical_not(np.round(Y))).astype(np.float32)

def PR(Y): #pos rate
    return pos(Y) / (pos(Y) + neg(Y))

def NR(Y): #neg rate
    return neg(Y) / (pos(Y) + neg(Y))

def TP(Y, Ypred): #true pos
    return np.sum(np.multiply(Y, np.round(Ypred))).astype(np.float32)

def FP(Y, Ypred): #false pos
    return np.sum(np.multiply(np.logical_not(Y), np.round(Ypred))).astype(np.float32)

def TN(Y, Ypred): #true neg
    return np.sum(np.multiply(np.logical_not(Y), np.logical_not(np.round(Ypred)))).astype(np.float32)

def FN(Y, Ypred): #false neg
    return np.sum(np.multiply(Y, np.logical_not(np.round(Ypred)))).astype(np.float32)

def FP_soft(Y, Ypred):
    return np.sum(np.multiply(np.logical_not(Y), Ypred)).astype(np.float32)

def FN_soft(Y, Ypred):
    return np.sum(np.multiply(Y, 1 - Ypred)).astype(np.float32)

#note: TPR + FNR = 1; TNR + FPR = 1
def TPR(Y, Ypred): #TP rate
    return 1-FNR(Y, Ypred)

def FPR(Y, Ypred): #FP rate
    negY = neg(Y)
    if negY == 0:
        return 0
    else:
        return FP(Y, Ypred) / negY

def TNR(Y, Ypred): #TN rate
    return TN(Y, Ypred) / neg(Y)

def FNR(Y, Ypred): #FP rate
    posY = pos(Y)
    if posY == 0:
        return 0
    else:
        return FN(Y, Ypred) / posY

def FPR_soft(Y, Ypred):
    return FP_soft(Y, Ypred) / neg(Y)

def FNR_soft(Y, Ypred):
    return FN_soft(Y, Ypred) / pos(Y)

def calibPosRate(Y, Ypred):
    return TP(Y, Ypred) / pos(Ypred)

def calibNegRate(Y, Ypred):
    return TN(Y, Ypred) / neg(Ypred)

def errRate(Y, Ypred):
    return (FP(Y, Ypred) + FN(Y, Ypred)) / float(Y.shape[0])

'''def accuracy(Y, Ypred):
    return 1 - errRate(Y, Ypred)'''

def accuracy(Y, Ypred):

    if Y.shape[1] > 1:
        acc = tf.keras.metrics.CategoricalAccuracy()
    else:
        acc = tf.keras.metrics.BinaryAccuracy()
    
    acc.update_state(y_true= Y, y_pred=Ypred)
    return acc.result().numpy()

def DI_FP(Y, Ypred, A, adim):
    #print('call di fp')
    if adim == 1:
        fpr1 = subgroup(FPR, A, Y, Ypred)
        fpr0 = subgroup(FPR, 1 - A, Y, Ypred)
        return abs(fpr1 - fpr0)

    group_difference = categorical_subgroup(fn=FPR, Amask=A, adim=adim, Y=Y, Ypred=Ypred)

    return (abs(group_difference))
    
def DI_TP(Y, Ypred, A, adim):
    #print('call di tp')
    if adim == 1:
        tpr1 = subgroup(TPR, A, Y, Ypred)
        tpr0 = subgroup(TPR, 1 - A, Y, Ypred)
        return abs(tpr1 - tpr0)

    group_difference = categorical_subgroup(fn=TPR, Amask=A, adim=adim, Y=Y, Ypred=Ypred)

    return (abs(group_difference))

def DI_FN(Y, Ypred, A):
    #print('call di fn')
    fnr1 = subgroup(FNR, A, Y, Ypred)
    fnr0 = subgroup(FNR, 1 - A, Y, Ypred)
    return abs(fnr1 - fnr0)

def DI_FP_soft(Y, Ypred, A):
    fpr1 = subgroup(FPR_soft, A, Y, Ypred)
    fpr0 = subgroup(FPR_soft, 1 - A, Y, Ypred)
    return abs(fpr1 - fpr0)

def DI_FN_soft(Y, Ypred, A):
    fnr1 = subgroup(FNR_soft, A, Y, Ypred)
    fnr0 = subgroup(FNR_soft, 1 - A, Y, Ypred)
    return abs(fnr1 - fnr0)

'''
SHOULD CONSIDER TRUE POSITIVES AND FALSE POSITIVES
def DI(Y, Ypred, A):
    #print('call di')
    return (DI_FN(Y, Ypred, A) + DI_FP(Y, Ypred, A)) * 0.5
'''
def DEqOdds(Y, Ypred, A, adim): #deltaEOdds
    #print('call di')
    return 1 - ((DI_TP(Y, Ypred, A, adim) + DI_FP(Y, Ypred, A, adim)) * 0.5)

''' CONSIDER THE TRUE POSITIVE FOR BOTH GROUPS
def DI_soft(Y, Ypred, A): #deltaEOpp
    return (DI_FN_soft(Y, Ypred, A) + DI_FP_soft(Y, Ypred, A)) * 0.5'''

def DEqOpp(Y, Ypred, A, adim): #deltaEOpp
    
    if adim == 1:
        tpr1 = subgroup(TPR, A, Y, Ypred)
        tpr0 = subgroup(TPR, 1 - A, Y, Ypred)
        return 1 - (abs(tpr1 - tpr0))

    group_difference = categorical_subgroup(fn=TPR, Amask=A, adim=adim, Y=Y, Ypred=Ypred)

    return 1 - (abs(group_difference))
    

def DP(Ypred, A, adim): #deltaDP
    if adim == 1:
        return 1 - (abs(subgroup(PR, A, Ypred) - subgroup(PR, 1 - A, Ypred)))

    group_difference = categorical_subgroup(fn=PR, Amask=A, adim=adim, Y=Ypred)

    return 1 - (abs(group_difference))

def categorical_subgroup(fn, Amask, adim, Y, Ypred=None, priviliged_idx=-1):
    # in our data prep for adult dataset 
    # the race order id 'race_ Amer-Indian-Eskimo','race_ Asian-Pac-Islander','race_ Black','race_ Other','race_ White'
    # we want to know the difference between 'race_ White' and the others
    
    groups_difference = []

    priviliged_result = subgroup(fn, Amask[:, priviliged_idx], Y, Ypred)
    
    if priviliged_idx == -1:
        idx_list = [idx for idx in range(adim)][:-1]
    else:
        idx_list = [idx for idx in range(adim) if idx != priviliged_idx]

    for group_idx in idx_list:
        group_result = subgroup(fn, Amask[:, group_idx], Y, Ypred)
        groups_difference.append(
            abs(priviliged_result - group_result)
        )
    #print(groups_difference)
    reduce_mean = sum(groups_difference)/len(groups_difference)
    
    return reduce_mean

    # previous categorical_subgroup implementation
    # group_difference = []
    # for group_idx in range(adim):
    #     if group_difference:
    #         group_difference -= subgroup(fn, Amask[:, group_idx], Y, Ypred)
    #     else:
    #         group_difference = subgroup(fn, Amask[:, group_idx], Y, Ypred)
    
    # return group_difference

def subgroup(fn, mask, Y, Ypred=None):
    #print('call subgroup')
    m = np.greater(mask, 0.5).flatten()
    #print(m[:5])
    Yf = Y.flatten()
    if not Ypred is None: #two-argument functions
        Ypredf = Ypred.flatten()
        #print('call function {}'.format(fn))
        return fn(Yf[m], Ypredf[m]) #access the indexes that are True (the True check subgroup)
    else: #one-argument functions
        return fn(Yf[m])

def NLL(Y, Ypred, eps=eps):
    return -np.mean(np.multiply(Y, np.log(Ypred + eps)) + np.multiply(1. - Y, np.log(1 - Ypred + eps)))

if __name__ == '__main__':
    main()