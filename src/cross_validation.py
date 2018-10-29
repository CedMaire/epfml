import numpy as np
from implementations import least_squares_GD, ridge_regression
from cost_computer import compute_loss
from label_predictor import predict_labels

def build_k_indices(y, k_fold, seed):
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, lambda_):
    # get k'th subgroup in test, others in train
    test_indice = k_indices[k]
    train_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    train_indice = train_indice.reshape(-1)
    y_test = y[test_indice]
    y_train = y[train_indice]
    x_test = x[test_indice]
    x_train = x[train_indice]

#    w, loss_train = least_squares_GD(y_train, x_train, np.asarray(np.zeros(len(x_train[0]))), 5, lambda_)
    w, loss_train = ridge_regression(y_train, x_train, lambda_)
#    loss_test = compute_loss(y_test, x_test, w)
    loss_test = np.sqrt(2 * compute_loss(y_test, x_test, w))
    y_pred = predict_labels(w, x_test)
    counter = 0
    for x in range(y_test.shape[0]):
        if y_pred[x] == y_test[x]:
            counter += 1
    percent = 100*counter/y_test.shape[0]
    return loss_train, loss_test, w, percent

def build_poly(tx, degree):
    for idx, x in enumerate(tx.T):
        if idx == 0:
            arr_out = build_poly_one_column(x, degree)
        else:
            arr_out = np.c_[arr_out, build_poly_one_column(x, degree)]
    return arr_out

def build_poly_one_column(x, degree):
    arr = np.zeros((x.shape[0], degree+1))
    for degre in range(degree+1):
        arr[:,degre] = np.power(x, degre)
    return arr

def cross_validation_search_param(y, tx, set_um):
    seed = 12
    k_fold = 4
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    #Best parameter
    best_w = []
    best_ratio = 0
    best_lambda = 0
    best_degree = -1

    lambda_range = ""
    degree_range = ""
    inc = 0.00001
    if set_um == 0:
        lambda_range = np.arange(inc, 0.0002 + inc, inc)
        degree_range = [5]
    elif set_um == 1:
        lambda_range = np.arange(0.0098, 0.0100 + inc, inc)
        degree_range = [11]
    elif set_um == 2:
        lambda_range = np.arange(0.2989, 0.2991 + inc, inc)
        degree_range = [-1]
    elif set_um == 3:
        lambda_range = np.arange(0.8997, 0.8999 + inc, inc)
        degree_range = [-1]
    elif set_um == 4:
        lambda_range = np.arange(0.0004, 0.0006 + inc, inc)
        degree_range = [13]
    elif set_um == 5:
        lambda_range = np.arange(0.0012, 0.0014 + inc, inc)
        degree_range = [12]
    elif set_um == 6:
        lambda_range = np.arange(0.0015, 0.0017 + inc, inc)
        degree_range = [11]
    elif set_um == 7:
        lambda_range = np.arange(0.0025, 0.0027 + inc, inc)
        degree_range = [13]

    #Test of different degrees
    count = 0
    for d in degree_range:
        print("\tDegree:", d)
        if d >= 0:
            tx_train = build_poly(tx, d)
        else:
            tx_train = tx
        for l in lambda_range:
            print("\t\tLambda:", l)
            # define lists to store the ratio of true mapping
            ratio = 0
            ratios= []
            for k in range(k_fold):
                _, _, w, ratio = cross_validation(y, tx_train, k_indices, k, l)
                ratios.append(ratio)
            if np.mean(ratios) > best_ratio:
                best_w = w
                best_lambda = l
                best_ratio = np.mean(ratios)
                best_degree = d
            count += 1

    return best_lambda, best_degree, best_ratio, best_w
