import numpy as np
import gradient_descent
from logistic_regression import *
from data_loader import load_data, DATA_PATH_TEST, DATA_PATH_TRAIN, DATA_PATH_SAMPLE_SUBMISSION_TEST
from implementations import *
from label_predictor import predict_labels
from csv_creator import create_csv
from cross_validation import *
from plots import print_stats, plot_label_feature_corrcoefs

if __name__ == "__main__":
    """
    y_train, tx_train, ids_train = load_data(DATA_PATH_TRAIN)
    tx_train = np.c_[y_train, tx_train]
    corrcoefs = []
    for i, col_i in enumerate(tx_train.T):
        if i > 1:
            cc = np.corrcoef(tx_train.T[0], col_i)[0][1]
            corrcoefs.append(cc)

    plot_label_feature_corrcoefs(np.linspace(0, 29, 30), corrcoefs, "Correlation Coefficient Between the Features and the Label", "Features", "Correlation Coefficient")
    """

    """
    y_tx = np.c_[y_train, tx_train]
    y_tx_one = y_tx[y_tx[:, 0] == 1, :]
    y_tx_minusone = y_tx[y_tx[:, 0] == -1, :]

    print("-------------------------------MEAN")
    y_tx_one_mean = np.mean(y_tx_one, axis=0)
    y_tx_minusone_mean = np.mean(y_tx_minusone, axis=0)
    mean_zip = list(zip(y_tx_one_mean, y_tx_minusone_mean))
    for idx, pair in enumerate(mean_zip):
        print(idx, pair)

    print("-------------------------------MEDIAN")
    y_tx_one_median = np.median(y_tx_one, axis=0)
    y_tx_minusone_median = np.median(y_tx_minusone, axis=0)
    median_zip = list(zip(y_tx_one_median, y_tx_minusone_median))
    for idx, pair in enumerate(median_zip):
        print(idx, pair)

    print("-------------------------------AVERAGE")
    y_tx_one_average = np.average(y_tx_one, axis=0)
    y_tx_minusone_average = np.average(y_tx_minusone, axis=0)
    average_zip = list(zip(y_tx_one_average, y_tx_minusone_average))
    for idx, pair in enumerate(average_zip):
        print(idx, pair)

    print("-------------------------------STD")
    y_tx_one_std = np.std(y_tx_one, axis=0)
    y_tx_minusone_std = np.std(y_tx_minusone, axis=0)
    std_zip = list(zip(y_tx_one_std, y_tx_minusone_std))
    for idx, pair in enumerate(std_zip):
        print(idx, pair)

    print("-------------------------------GRADIENT")
    w, loss = ridge_regression(y_train, tx_train, 0.037)
    w_sum = np.sum(np.abs(w))
    w = list(map(lambda x: (x, 100 * np.abs(x) / w_sum), w))
    print(loss)
    for idx, pair in enumerate(w):
        print(idx, pair)
    """

    print("Train")
    ys_train, txs_train, ids_train = load_data(DATA_PATH_TRAIN)

    for y_train, tx_train, id_train in zip(ys_train, txs_train, ids_train):
        print(y_train.shape)
        print(y_train)
        print(tx_train)
        print(id_train)
        print("--------------------------------------------")

#    tx_train = np.c_[y_train, tx_train]
#    tx_train = np.delete(tx_train, [5, 12, 15, 18, 19, 20, 21, 23, 25, 27, 28, 29, 30], axis=1)
#    txs_train = build_poly(txs_train, 3)
#    w, loss = gradient_descent.test_GD(y_train, tx_train)
#    w, loss = gradient_descent.test_SGD(y_train, tx_train)
#    w, loss = least_squares(y_train, tx_train)
#    w, loss = ridge_regression(y_train, tx_train, 0.037)
#    w, loss = logistic_regression_gradient_descent(y_train, tx_train, 0.9999, 3000)

#    w, loss_train = logistic_regression.regularized_logistic_regression_gradient_descent(y_train, tx_train, 0.1, 1000,0.3)
    
#    print(loss_train)
#    print(w)

    print("Test")
    ys_test, txs_test, ids_test = load_data(DATA_PATH_TEST)

    for y_test, tx_test, id_test in zip(ys_test, txs_test, ids_test):
        print(y_test)
        print(tx_test)
        print(id_test)
        print("--------------------------------------------")

#    tx_test = np.c_[y_test, tx_test]
#    tx_test = np.delete(tx_test, [5, 12, 15, 18, 19, 20, 21, 23, 25, 27, 28, 29, 30], axis=1)
#    tx_test = build_poly(tx_test, 3)
#    y_pred = predict_labels(w, tx_test)
#    print_stats(y_pred)

#    create_csv(ids_test, y_pred, DATA_PATH_SAMPLE_SUBMISSION_TEST)
