import numpy as np
import gradient_descent
from data_loader import load_data, DATA_PATH_TEST, DATA_PATH_TRAIN, DATA_PATH_SAMPLE_SUBMISSION

if __name__ == "__main__":
    y, tx, ids = load_data(DATA_PATH_TRAIN)
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

#    print("IDs")
#    print(ids)
#    print("--------------------------------------------------")
#    print("y")
#    print(y)
#    print("--------------------------------------------------")
#    print("TX")
#    print(tx)
#    print(tx.shape)

    print("-------------------------------GRADIENT")
    w = gradient_descent.test_GD(y, tx)
    print(w)

#    print(y, len(y))
#    print(tx, len(tx))
#    print(ids, len(ids))
#    print("--------------------------------------------------")
#    print(np.mean(tx, axis=0))
#    print(np.std(tx, axis=0))
#    print("--------------------------------------------------")
#    print(y.shape)
#    print(tx.shape)
