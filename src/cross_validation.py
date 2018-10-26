
# coding: utf-8

# In[22]:


import numpy as np
from data_loader import load_data, DATA_PATH_TEST, DATA_PATH_TRAIN, DATA_PATH_SAMPLE_SUBMISSION_TEST
import logistic_regression
import gradient_descent
import matplotlib.pyplot as plt
from label_predictor import predict_labels


# In[2]:


y, tx, ids_train = load_data(DATA_PATH_TRAIN)


# In[3]:


def build_k_indices(y, k_fold, seed):
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


# In[30]:


def cross_validation(y, x, k_indices, k):
    # get k'th subgroup in test, others in train
    test_indice = k_indices[k]
    train_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    train_indice = train_indice.reshape(-1)
    y_test = y[test_indice]
    y_train = y[train_indice]
    x_test = x[test_indice]
    x_train = x[train_indice]
    
    w, loss_train = logistic_regression.regularized_logistic_regression_gradient_descent(y_train, x_train, 0.1, 1000,0.3)
    loss_test = logistic_regression.calculate_loss(y_test, x_test, w)
    y_pred = predict_labels(w, x_test)
    counter = 0
    y_pred = [0 if x==-1 else x for x in y_pred]
    for x in range(y_test.shape[0]):
        if y_pred[x] == y_test[x]:
            counter += 1
    print(100*counter/y_test.shape[0])
    return loss_train, loss_test,w


# In[9]:


def build_poly(tx, degree):
    for idx, x in enumerate(tx.T):
        if idx == 0:
            arr_out = build_poly_one_column(x, degree)
        else:
            arr_out = np.c_[arr_out, build_poly_one_column(x, degree)]
    return arr_out


# In[10]:


def build_poly_one_column(x, degree):
    arr = np.zeros((x.shape[0], degree+1))
    for degre in range(degree+1):
        arr[:,degre] = np.power(x, degre)
    return arr


# In[31]:


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

