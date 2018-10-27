
# coding: utf-8

# In[1]:

import time
import numpy as np
from data_loader import load_data, DATA_PATH_TEST, DATA_PATH_TRAIN, DATA_PATH_SAMPLE_SUBMISSION_TEST
import logistic_regression
import gradient_descent
import matplotlib.pyplot as plt
from label_predictor import predict_labels


# In[2]:

#print("Loading Data...")
#y, tx, ids_train = load_data(DATA_PATH_TRAIN)


# In[3]:


def build_k_indices(y, k_fold, seed):
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


# In[44]:


def cross_validation(y, x, k_indices, k, gamma, alpha):
    # get k'th subgroup in test, others in train
    test_indice = k_indices[k]
    train_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    train_indice = train_indice.reshape(-1)
    y_test = y[test_indice]
    y_train = y[train_indice]
    x_test = x[test_indice]
    x_train = x[train_indice]
    
    w, loss_train = logistic_regression.regularized_logistic_regression_gradient_descent(y_train, x_train, gamma, 1000, alpha)
    loss_test = logistic_regression.calculate_loss(y_test, x_test, w)
    y_pred = predict_labels(w, x_test)
    counter = 0
    y_pred = [0 if x==-1 else x for x in y_pred]
    for x in range(y_test.shape[0]):
        if y_pred[x] == y_test[x]:
            counter += 1
    percent = 100*counter/y_test.shape[0]
#     print(percent)
    return loss_train, loss_test, w, percent


# In[5]:


def build_poly(tx, degree):
    for idx, x in enumerate(tx.T):
        if idx == 0:
            arr_out = build_poly_one_column(x, degree)
        else:
            arr_out = np.c_[arr_out, build_poly_one_column(x, degree)]
    return arr_out


# In[6]:


def build_poly_one_column(x, degree):
    arr = np.zeros((x.shape[0], degree+1))
    for degre in range(degree+1):
        arr[:,degre] = np.power(x, degre)
    return arr


# In[22]:


def cross_validation_demo():
    seed = 12
    k_fold = 4
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    loss_tr = []
    loss_te = []
    # cross validation
    loss_tr_tmp = []
    loss_te_tmp = []
    tx_train = np.delete(tx, [5, 12, 15, 18, 19, 20, 21, 23, 25, 27, 28, 29, 30], axis=1)
    tx_train = build_poly(tx_train, 3)
    for k in range(k_fold):
        loss_tr, loss_te,_ = cross_validation(y, tx_train, k_indices, k)
        loss_tr_tmp.append(loss_tr)
        loss_te_tmp.append(loss_te)
        print("the loss of the training set is: ", np.mean(loss_tr_tmp))
        print("the loss of the test set is: ", np.mean(loss_te_tmp))

#cross_validation_demo()


# In[45]:


def cross_validation_search_param(tx, y, set_um):
    seed = 12
    k_fold = 4
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    #Best parameter
    bestW = []
    bestRatio = 0
    bestG = 0
    bestA = 0
    bestD = 0

    gamma_range = ""
    alpha_range = ""
    degree_range = ""
    if set_um == 0:
#        gamma_range = np.arange(0.4, 0.61, 0.01)
#        alpha_range = np.arange(0.4, 0.61, 0.01)
#        degree_range = np.arange(2, 3, 1)
        gamma_range = [0.4]
        alpha_range = [0.4]
        degree_range = [2]
    elif set_um == 1:
#        gamma_range = np.arange(0, 0.21, 0.01)
#        alpha_range = np.arange(0, 0.21, 0.01)
#        degree_range = np.arange(2, 3, 1)
        gamma_range = [0.08]
        alpha_range = [0.0]
        degree_range = [2]
    elif set_um == 2:
#        gamma_range = np.arange(0.4, 0.61, 0.01)
#        alpha_range = np.arange(0, 0.21, 0.01)
#        degree_range = np.arange(2, 3, 1)
        gamma_range = [0.5]
        alpha_range = [0.1]
        degree_range = [2]
    elif set_um == 3:
#        gamma_range = np.arange(0.5, 0.71, 0.01)
#        alpha_range = np.arange(0, 0.21, 0.01)
#        degree_range = np.arange(2, 3, 1)
        gamma_range = [0.6]
        alpha_range = [0.1]
        degree_range = [2]
    elif set_um == 4:
#        gamma_range = np.arange(0.1, 0.21, 0.01)
#        alpha_range = np.arange(0.7, 0.81, 0.01)
#        degree_range = np.arange(2, 3, 1)
        gamma_range = [0.15]
        alpha_range = [0.7]
        degree_range = [2]
    elif set_um == 5:
#        gamma_range = np.arange(0, 0.21, 0.01)
#        alpha_range = np.arange(0, 0.21, 0.01)
#        degree_range = np.arange(2, 3, 1)
        gamma_range = [0.09]
        alpha_range = [0.0]
        degree_range = [2]
    elif set_um == 6:
#        gamma_range = np.arange(0, 0.21, 0.01)
#        alpha_range = np.arange(0.2, 0.41, 0.01)
#        degree_range = np.arange(2, 3, 1)
        gamma_range = [0.06]
        alpha_range = [0.2]
        degree_range = [2]
    elif set_um == 7:
#        gamma_range = np.arange(0, 1.1, 0.1)
#        alpha_range = np.arange(0, 1.1, 0.1)
#        degree_range = np.arange(2, 3, 1)
        gamma_range = [0.8]
        alpha_range = [0.30000000000000004]
        degree_range = [2]

    #Test of different degrees
    count = 0
    for d in degree_range:  
        tx_train = build_poly(tx, d)
        for a in alpha_range:
            for g in gamma_range:
                # define lists to store the ratio of true mapping
                ratio = 0
                ratios= []
                for k in range(k_fold):
                    _, _, w, ratio = cross_validation(y, tx_train, k_indices, k, g, a)
                    ratios.append(ratio)
                if np.mean(ratios) > bestRatio:
                    bestW = w
                    bestG = g
                    bestA = a
                    bestD = d
                    bestRatio = np.mean(ratios)
                count += 1
#                 if(count%100 == 0):
#                     print(count)
#                 print("ratio:", np.mean(ratios))
    print("bestRatio:", bestRatio)
    return bestW, bestG, bestA, bestD

#counter = 0
#for y_test, tx_test, id_test in zip(y, tx, ids_train):
##    if counter != 7:
##        counter += 1
##        continue
#    print("Start set:", counter)
#    start = time.time()
#    bestW, bestG, bestA, bestD = cross_validation_search_param(tx_test, y_test, counter)
#    end = time.time()
#    print("Time:", end - start)
#    print("Set ", counter)
#    print("bestW ", list(bestW))
#    print("bestG ", bestG)
#    print("bestA ", bestA)
#    print("bestD ", bestD)
#    print("--------------------------------------------------------------------------------")
#    counter += 1
