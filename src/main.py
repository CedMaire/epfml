import numpy as np
import gradient_descent
import logistic_regression
from data_loader import load_data, DATA_PATH_TEST, DATA_PATH_TRAIN, DATA_PATH_SAMPLE_SUBMISSION

if __name__ == "__main__":
    y, tx, ids = load_data(DATA_PATH_TRAIN)
#    logistic_regression.logistic_regression_gradient_descent(y, tx, 0.0000001, 1000)
    logistic_regression.regularized_logistic_regression_gradient_descent(y, tx, 0.0000001, 1000,0.01)
#    gradient_descent.test_GD(y, tx)
#    gradient_descent.test_SGD(y, tx)
#    print(y, len(y))
#    print(tx, len(tx))
#    print(ids, len(ids))
#    print("--------------------------------------------------")
#    print(np.mean(tx, axis=0))
#    print(np.std(tx, axis=0))
#    print("--------------------------------------------------")
#    print(y.shape)
#    print(tx.shape)