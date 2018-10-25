import numpy as np
import gradient_descent
from logistic_regression import *
from data_loader import load_data, DATA_PATH_TEST, DATA_PATH_TRAIN, DATA_PATH_SAMPLE_SUBMISSION_TEST
from implementations import *
from label_predictor import predict_labels
from csv_creator import create_csv

if __name__ == "__main__":
    """
    y_train, tx_train, ids_train = load_data(DATA_PATH_TRAIN)
    tx_train = np.c_[y_train, tx_train]
#    tx_train = np.delete(tx_train, [4, 6, 8, 9, 10, 11, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], axis=1)

    corrcoefs = []
    for i, col_i in enumerate(tx_train.T):
        for j, col_j in enumerate(tx_train.T):
            if i < j and i != 1 and j != 1:
                corrcoefs.append((i, j, np.corrcoef(col_i, col_j)[0][1]))

    corrcoefs = list(map(lambda triple: triple[1], filter(lambda triple: triple[0] == 0 and abs(triple[2]) < 0.16, corrcoefs)))
    print(corrcoefs)
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
    y_train, tx_train, ids_train = load_data(DATA_PATH_TRAIN)
    tx_train = np.c_[y_train, tx_train]
    tx_train = np.delete(tx_train, [0, 4, 6, 8, 9, 10, 11, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], axis=1)

#    w, loss = gradient_descent.test_GD(y_train, tx_train)
#    w, loss = gradient_descent.test_SGD(y_train, tx_train)
#    w, loss = least_squares(y_train, tx_train)
#    w, loss = ridge_regression(y_train, tx_train, 0.037)
    w, loss = logistic_regression_gradient_descent(y_train, tx_train, 0.9999, 1000)
#    w, loss = regularized_logistic_regression_gradient_descent(y_train, tx_train, 0.0000001, 1000, 0.01)

    print(loss)
    print(w)

    print("Test")
    y_test, tx_test, ids_test = load_data(DATA_PATH_TEST)
    tx_test = np.c_[y_test, tx_test]
    tx_test = np.delete(tx_test, [0, 4, 6, 8, 9, 10, 11, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], axis=1)

    y_pred = predict_labels(w, tx_test)

    y_len = len(y_pred)
    y_pos = len(y_pred[y_pred == 1])
    y_neg = len(y_pred[y_pred == -1])
    print(" 1 ->", y_pos / y_len)
    print("-1 ->", y_neg / y_len)

    create_csv(ids_test, y_pred, DATA_PATH_SAMPLE_SUBMISSION_TEST)

#    print("Print")
#    for i in range(1000):
#        print(y_pred[i])
