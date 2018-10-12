import numpy as np
import data_loader

if __name__ == "__main__":
    y, tx, ids = data_loader.load_data(data_loader.DATA_PATH_TRAIN)

    print(y, len(y))
    print(tx, len(tx))
    print(ids, len(ids))
    print("--------------------------------------------------")
    print(np.mean(tx, axis=0))
    print(np.std(tx, axis=0))
    print("--------------------------------------------------")
    print(y.shape)
    print(tx.shape)
