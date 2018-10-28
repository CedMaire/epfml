import numpy as np
from data_loader import load_data, DATA_PATH_TRAIN
from implementations import least_squares_GD, least_squares_SGD, least_squares, ridge_regression, logistic_regression, reg_logistic_regression
from label_predictor import predict_labels
from cross_validation import build_poly
def compute_correctness(y, tx, w):
    y_pred = predict_labels(w, tx)
    y[np.where(y == 0)] = -1

    return len(np.where(y - y_pred == 0)[0]) / len(y)

def print_stats(loss, correctness):
    print("\t\t\tloss:\t\t%.2f" % loss)
    print("\t\t\tcorrectness:\t%.2f" % correctness)

if __name__ == "__main__":
    print("Data loading...")
    ys_train, txs_train, ids_train = load_data(DATA_PATH_TRAIN)

    print("Computing ws...")
    for index, model in enumerate(zip(ys_train, txs_train, ids_train)):
        print("\tModel ", index)
       
        y_train, tx_train, id_train = model[0], model[1], model[2]
        
        print("\t\tleast_squares_GD")
        w, loss = least_squares_GD(y_train, tx_train, np.asarray(np.zeros(len(tx_train[0]))), 1000, 0.2)
        print_stats(loss, compute_correctness(y_train, tx_train, w))

        print("\t\tleast_squares_SGD")
        w, loss = least_squares_SGD(y_train, tx_train, np.asarray(np.zeros(len(tx_train[0]))), 1000, 0.005)
        print_stats(loss, compute_correctness(y_train, tx_train, w))

        print("\t\tleast_squares")
        w, loss = least_squares(y_train, tx_train)
        print_stats(loss, compute_correctness(y_train, tx_train, w))
        tx_train = build_poly(tx_train, 2)
        print("\t\tridge_regression")
        w, loss = ridge_regression(y_train, tx_train, 0.1)
        print_stats(loss, compute_correctness(y_train, tx_train, w))

        print("\t\tlogistic_regression")
        w, loss = logistic_regression(y_train, tx_train, np.asarray(np.zeros(len(tx_train[0]))), 1000, 0.9)
        print_stats(loss, compute_correctness(y_train, tx_train, w))
        
        print("\t\treg_logistic_regression")
        w, loss = reg_logistic_regression(y_train, tx_train, 0.1, np.asarray(np.zeros(len(tx_train[0]))), 1000, 0.9)
        print_stats(loss, compute_correctness(y_train, tx_train, w))
