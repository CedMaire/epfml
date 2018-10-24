import data_loader
from implementations import *

if __name__ == "__main__":
    yb, input_data, ids = data_loader.load_data(data_loader.DATA_PATH_TRAIN)

    print(yb, len(yb))
    print(input_data, len(input_data))
    print(ids, len(ids))
    
    w, loss = least_squares(yb, input_data)
    print(w, loss)
    w, loss = ridge_regression(yb, input_data, 0.037)
    print(w, loss)
