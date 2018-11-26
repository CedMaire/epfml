import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

def model_linear_logistic_regression(Y, X):
    """
    Generates a linear model using linear regression.

    (Function greatly inspired from the given file "segment_aerial_images.ipynb" given by the professor.)

    :param Y: the vector of groundtruth labels
    :param X: the matrix of features
    """

    logistic_regression = linear_model.LogisticRegression(C=1e5, class_weight="balanced")
    logistic_regression.fit(X, Y)

    prediction = logistic_regression.predict(X)

    prediction_n = np.nonzero(prediction)[0]
    Y_n = np.nonzero(Y)[0]

    true_positive_rate = len(list(set(Y_n) & set(prediction_n))) / float(len(prediction))
    print("True positive rate = " + str(true_positive_rate))

    plt.scatter(X[:, 0], X[:, 1], c=prediction, edgecolors='k', cmap=plt.cm.Paired)
    plt.show()

    return logistic_regression
