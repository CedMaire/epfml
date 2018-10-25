import numpy as np
import gradient_descent
from logistic_regression import *
from data_loader import load_data, DATA_PATH_TEST, DATA_PATH_TRAIN, DATA_PATH_SAMPLE_SUBMISSION_TEST
from implementations import *
from label_predictor import predict_labels
from csv_creator import create_csv

if __name__ == "__main__":
    y_train, tx_train, ids_train = load_data(DATA_PATH_TRAIN)


    y_tx = np.c_[y, tx]
    y_tx_one = y_tx[y_tx[:, 0] == 1, :]
    y_tx_minusone = y_tx[y_tx[:, 0] == -1, :]

    print("-------------------------------MEAN")
    y_tx_one_mean = np.mean(y_tx_one, axis=0)
    y_tx_minusone_mean = np.mean(y_tx_minusone, axis=0)
    mean_zip = list(zip(y_tx_one_mean, y_tx_minusone_mean))
    print(mean_zip)

    print("-------------------------------MEDIAN")
    y_tx_one_median = np.median(y_tx_one, axis=0)
    y_tx_minusone_median = np.median(y_tx_minusone, axis=0)
    median_zip = list(zip(y_tx_one_median, y_tx_minusone_median))
    print(median_zip)

    print("-------------------------------AVERAGE")
    y_tx_one_average = np.average(y_tx_one, axis=0)
    y_tx_minusone_average = np.average(y_tx_minusone, axis=0)
    average_zip = list(zip(y_tx_one_average, y_tx_minusone_average))
    print(average_zip)

    print("-------------------------------STD")
    y_tx_one_std = np.std(y_tx_one, axis=0)
    y_tx_minusone_std = np.std(y_tx_minusone, axis=0)
    std_zip = list(zip(y_tx_one_std, y_tx_minusone_std))
    print(std_zip)

    print("-------------------------------GRADIENT")
    w = gradient_descent.test_GD(y, tx)
    print(w)

"""SUBMISSION
    print("Train")
    y_train, tx_train, ids_train = load_data(DATA_PATH_TRAIN)
#    w, loss = gradient_descent.test_GD(y_train, tx_train)
#    w, loss = gradient_descent.test_SGD(y_train, tx_train)
#    w, loss = least_squares(y_train, tx_train)
    w, loss = ridge_regression(y_train, tx_train, 0.037)
#    w, loss = logistic_regression_gradient_descent(y_train, tx_train, 0.0000001, 1000)
#    w, loss = regularized_logistic_regression_gradient_descent(y_train, tx_train, 0.0000001, 1000, 0.01)

    print(loss)
    print(w)

    print("Test")
    _, tx_test, ids_test = load_data(DATA_PATH_TEST)
    y_pred = predict_labels(w, tx_test)

    create_csv(ids_test, y_pred, DATA_PATH_SAMPLE_SUBMISSION_TEST)

#    print("Print")
#    for i in range(1000):
#        print(y_pred[i])
"""
