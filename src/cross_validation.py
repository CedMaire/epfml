import numpy as np
from implementations import ridge_regression
from cost_computer import compute_loss
from label_predictor import predict_labels
from data_loader import load_data, DATA_PATH_TRAIN

def build_k_indices(y, number_of_subset, seed):
    """
    Generates the random indices for cross-validation test and train subsets.

    :param y: expected labels vector
    :param number_of_subset: number of subsets
    :param seed: seed for randomness
    :returns: k_indices - indices for test and train subset
    """

    rows_num = len(y)
    inter = int(rows_num / number_of_subset)

    #set the random seed
    np.random.seed(seed)
    indices = np.random.permutation(rows_num)

    subset_indices = []
    for i in range(number_of_subset):
        subset_indices.append(indices[i * inter: (i + 1) * inter])

    return np.array(subset_indices)

def cross_validation(y, x, k_indices, k, lambda_):
    """
    Cross validates the the ridge regression algorithm for a given lambda.

    :param y: expected labels vector
    :param x: data matrix
    :param k_indices: indices for test and train subset
    :param k: number of cross-validation subsets
    :param lambda_: lambda hyper-parameter for ridge regression
    :returns: loss_train - loss over the train set
              loss_test - loss over the test set
              ratio - ratio of correctness
    """

    #get the indices of the subsets
    test_set = k_indices[k]
    train_set = k_indices[~(np.arange(k_indices.shape[0]) == k)].reshape(-1)
    
    #get the subsets
    tx_train = x[train_set]
    y_train = y[train_set]
    tx_test = x[test_set]
    y_test = y[test_set]

    w, loss_train = ridge_regression(y_train, tx_train, lambda_)
    loss_test = np.sqrt(2 * compute_loss(y_test, tx_test, w))
    y_pred = predict_labels(w, tx_test)

    counter = 0
    for i in range(y_test.shape[0]):
        if y_pred[i] == y_test[i]:
            counter += 1

    return loss_train, loss_test, w, 100 * counter / y_test.shape[0]

def build_poly(tx, degree):
    """
    Builds the polynomial expension of a data set.

    :param tx: the data set
    :param degree: maximal degree
    :returns: polynomial expension of tx
    """

    for idx, x in enumerate(tx.T):
        if idx == 0:
            arr_out = build_poly_one_column(x, degree)
        else:
            arr_out = np.c_[arr_out, build_poly_one_column(x, degree)]

    return arr_out

def build_poly_one_column(x, degree):
    """
    Builds the polynomial expension of one feature column.

    :param x: vector
    :param degree: the maximal degree
    :returns: polynomial expension of x
    """

    arr = np.zeros((x.shape[0], degree+1))

    for degre in range(degree+1):
        arr[:,degre] = np.power(x, degre)

    return arr

def cross_validation_search_param(y, tx, set_um):
    """
    Function to find the best hyper-parameter lambda for the ridge regression algorithm.

    :param y: expected labels vector
    :param tx: matrix of data set
    :param set_num: index of the current subset
    :returns: best_lambda - best lambda hyper-parameter for ridge regression
              best_degree - best degree for the build_poly function
              best_ratio - best achieved ratio
              best_w - best generated weights
    """

    seed = 12
    k_fold = 4
    k_indices = build_k_indices(y, k_fold, seed)

    best_w = []
    best_ratio = 0
    best_lambda = 0
    best_degree = -1

    lambda_range = ""
    degree_range = ""
    inc = 0.00001

    # Definition of the lambda and degree ranges per subset.
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

    # Iterate over the ranges to get the best parameters.
    count = 0
    for d in degree_range:
        print("\tDegree:", d)

        if d >= 0:
            tx_train = build_poly(tx, d)
        else:
            tx_train = tx

        for l in lambda_range:
            print("\t\tLambda:", l)

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

# Main Function
if __name__ == "__main__":
    print("Loading train data...")
    ys_train, txs_train, ids_train = load_data(DATA_PATH_TRAIN)
    total_samples = np.sum(list(map(lambda ids: len(ids), ids_train)))

    print("Training...")
    best_lambdas = []
    best_degrees = []
    best_ratios = []
    best_ws = []
    for index, model in enumerate(zip(ys_train, txs_train, ids_train)):
        print("Set:", index)
        y_train, tx_train, id_train = model[0], model[1], model[2]
        model_samples = len(id_train)

        best_lambda, best_degree, best_ratio, best_w = cross_validation_search_param(y_train, tx_train, index)
        best_lambdas.append(best_lambda)
        best_degrees.append(best_degree)
        best_ratios.append((model_samples / total_samples, best_ratio))
        best_ws.append(best_w)

    for l, d, r, w in zip(best_lambdas, best_degrees, best_ratios, best_ws):
        print("-------------------------------------------")
        print("lambda:", l, "degree:", d, "ratio", r[1])
        print(list(w))

    print("-------------------------------------------")
    print("Correctness:", np.sum(np.asarray(list(map(lambda pair: pair[0] * pair[1], best_ratios)))))
