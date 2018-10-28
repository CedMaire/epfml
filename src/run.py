"""
def print_stats(y_pred):
    y_len = len(y_pred)
    print(" 1 ->", len(y_pred[y_pred == 1]) / y_len)
    print("-1 ->", len(y_pred[y_pred == -1]) / y_len)

ys_test, txs_test, ids_test = load_data(DATA_PATH_TEST)
ws = "???"

y_pred = np.asarray([[0, 0]])
for y_test, tx_test, id_test, w in zip(ys_test, txs_test, ids_test, ws):
    tx_test = build_poly(tx_test, 2)
    y_pred = np.vstack((y_pred, np.c_[id_test, predict_labels(w, tx_test)]))

y_pred = y_pred[1:,:]
y_pred = y_pred[y_pred[:,0].argsort()]

print_stats(y_pred[:, 1])
create_csv(y_pred[:, 0], y_pred[:, 1], DATA_PATH_SAMPLE_SUBMISSION_TEST)
"""
