import numpy as np
from data_loader import load_data, DATA_PATH_TRAIN, DATA_PATH_TEST, DATA_PATH_SAMPLE_SUBMISSION_TEST
from implementations import ridge_regression
from label_predictor import predict_labels
from cross_validation import build_poly
from csv_creator import create_csv

# Helper Functions
def print_stats(y_pred):
    """
    Prints the percentage of labels inside a prediction vector.

    :param y_pred: the prediction vector
    """

    y_len = len(y_pred)

    print(" 1 ->", len(y_pred[y_pred == 1]) / y_len)
    print("-1 ->", len(y_pred[y_pred == -1]) / y_len)

# Main Function
if __name__ == "__main__":
    print("Loading train data...")
    ys_train, txs_train, ids_train = load_data(DATA_PATH_TRAIN)

    print("Training...")
    ws = []
    for y_train, tx_train, id_train in zip(ys_train, txs_train, ids_train):
#        tx_train = build_poly(tx_train, 2)
        w, loss = ridge_regression(y_train, tx_train, 0.1)
        ws.append(w)

    print("Loading test data...")
    ys_test, txs_test, ids_test = load_data(DATA_PATH_TEST)
    y_pred = np.asarray([[0, 0]])

    print("Predicting...")
    for y_test, tx_test, id_test, w in zip(ys_test, txs_test, ids_test, ws):
#        tx_test = build_poly(tx_test, 2)
        y_pred = np.vstack((y_pred, np.c_[id_test, predict_labels(w, tx_test)]))

    print("Stats:")
    y_pred = y_pred[1:,:]
    y_pred = y_pred[y_pred[:,0].argsort()]
    print_stats(y_pred[:, 1])

    print("Creating CSV file...")
    create_csv(y_pred[:, 0], y_pred[:, 1], DATA_PATH_SAMPLE_SUBMISSION_TEST)

    print("Done!")
