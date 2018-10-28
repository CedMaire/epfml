import csv
import numpy as np

"""
Functions related to the creation of a CSV file containing the final results of the Machine Learning process.
"""

def create_csv(ids, y_pred, name):
    """
    Creates a CSV file for the given predictions (this file can be uploaded to Kaggle as a submission).

    (Function greatly inspired from the given file "proj1_helpers.py" given by the professor.)

    :param ids: the ids of the samples
    :param y_pred: the predictions for each sample
    :param name: the name of the file to create
    """

    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]

        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()

        for r1, r2 in zip(ids, y_pred):
            writer.writerow({
                                "Id": int(r1),
                                "Prediction": int(r2)
                            })
