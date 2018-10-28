from data_loader import load_data, DATA_PATH_TRAIN

if __name__ == "__main__":
    print("Data loading...")
    ys_train, txs_train, ids_train = load_data(DATA_PATH_TRAIN)

    for y_train, tx_train, id_train in zip(ys_train, txs_train, ids_train):
        pass
