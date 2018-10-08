import data_loader

if __name__ == "__main__":
    yb, input_data, ids = data_loader.load_data(data_loader.DATA_PATH_TRAIN)

    print(yb, len(yb))
    print(input_data, len(input_data))
    print(ids, len(ids))
