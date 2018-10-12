import numpy as np
import gradient_descent
from data_loader import load_data, DATA_PATH_TEST, DATA_PATH_TRAIN, DATA_PATH_SAMPLE_SUBMISSION

if __name__ == "__main__":
    y, tx, ids = load_data(DATA_PATH_TRAIN)

    gradient_descent.test_GD(y, tx)

#    print(y, len(y))
#    print(tx, len(tx))
#    print(ids, len(ids))
#    print("--------------------------------------------------")
#    print(np.mean(tx, axis=0))
#    print(np.std(tx, axis=0))
#    print("--------------------------------------------------")
#    print(y.shape)
#    print(tx.shape)
