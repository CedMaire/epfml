import csv
import numpy as np

# Function greatly inspired from the given file "proj1_helpers.py".
def create_csv(ids, y_pred, name):
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]

        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()

        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1),
                             "Prediction": int(r2)})
