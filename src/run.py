import numpy as np
from data_loader import load_data, DATA_PATH_TRAIN, DATA_PATH_TEST, DATA_PATH_SAMPLE_SUBMISSION_TEST
from label_predictor import predict_labels
from cross_validation import build_poly
from csv_creator import create_csv
from implementations import ridge_regression

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

    print("Loading test data...")
    ys_test, txs_test, ids_test = load_data(DATA_PATH_TEST)

    degrees = [5, 11, -1, -1, 13, 12, 11, 13]
    lambdas = [1e-05, 0.009899999999999996, 0.2989, 0.89972,
                0.0005000000000000002, 0.0013000000000000002, 0.0016000000000000003, 0.0026000000000000003]
    y_pred = np.asarray([[0, 0]])

    print("Predicting...")
    for index, model in enumerate(zip(ys_test, txs_test, ids_test)):
        y_test, tx_test, id_test = model[0], model[1], model[2]

        tx_train = txs_train[index]
        if degrees[index] >= 0:
            tx_test = build_poly(tx_test, degrees[index])
            tx_train = build_poly(tx_train, degrees[index])

        w, _ = ridge_regression(ys_train[index], tx_train, lambdas[index])
        y_pred = np.vstack((y_pred, np.c_[id_test, predict_labels(w, tx_test)]))

    print("Stats:")
    y_pred = y_pred[1:,:]
    y_pred = y_pred[y_pred[:,0].argsort()]
    print_stats(y_pred[:, 1])

    print("Creating CSV file...")
    create_csv(y_pred[:, 0], y_pred[:, 1], DATA_PATH_SAMPLE_SUBMISSION_TEST)

    print("Done!")
