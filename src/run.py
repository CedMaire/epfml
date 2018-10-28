import numpy as np
from data_loader import load_data, DATA_PATH_TRAIN, DATA_PATH_TEST, DATA_PATH_SAMPLE_SUBMISSION_TEST
from implementations import ridge_regression
from label_predictor import predict_labels
from cross_validation import build_poly
from csv_creator import create_csv
def print_stats(y_pred):
    y_len = len(y_pred)
    print(" 1 ->", len(y_pred[y_pred == 1]) / y_len)
    print("-1 ->", len(y_pred[y_pred == -1]) / y_len)

ys_train, txs_train, ids_train = load_data(DATA_PATH_TRAIN)
ws = []  
for y_train, tx_train, id_train in zip(ys_train, txs_train, ids_train):       
#         tx_train = build_poly(tx_train, 2)
        w, loss = ridge_regression(y_train, tx_train, 0.1)
        ws.append(w)
        
ys_test, txs_test, ids_test = load_data(DATA_PATH_TEST)        
y_pred = np.asarray([[0, 0]])
for y_test, tx_test, id_test, w in zip(ys_test, txs_test, ids_test, ws):
#     tx_test = build_poly(tx_test, 2)
    y_pred = np.vstack((y_pred, np.c_[id_test, predict_labels(w, tx_test)]))

    
y_pred = y_pred[1:,:]
y_pred = y_pred[y_pred[:,0].argsort()]
print_stats(y_pred[:, 1])

create_csv(y_pred[:, 0], y_pred[:, 1], DATA_PATH_SAMPLE_SUBMISSION_TEST)
