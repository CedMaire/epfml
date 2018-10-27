import numpy as np

"""
Data Cleaning:
    * Remove unwanted, duplicate or irrelevant results
    * Filter unwanted outliers
    * Handle Missing data? (Replace -999 by the mean of the remaining?)
"""

UNDEFINED_VALUE = -999

def remove_undefined_columns(x):
    bool_table = (x == UNDEFINED_VALUE)
    idx = bool_table.any(axis=0)

    return x[:, ~idx]

def replace_undefined_values(x):
    tx_T = x.T
    for col in tx_T:
#        values, counts = np.unique(col[col != UNDEFINED_VALUE], return_counts=True)
#        ind = np.argmax(counts)
#        col[np.where(col == UNDEFINED_VALUE)] = values[ind]
#        median = np.median(col[col != UNDEFINED_VALUE])
#        col[np.where(col == UNDEFINED_VALUE)] = median
        col[np.where(col == UNDEFINED_VALUE)] = 0

    return tx_T.T

def remove_samples(x):
    x = x[np.min(x, axis=1) != UNDEFINED_VALUE,:]
    return x

def split_data_meaningfuly(y, x, ids):
    x = np.c_[ids, y, x]

    # All the different interesting columns.
    col_DER_mass_MMC = 2
    col_PRI_jet_num = 24
    col_set_less_1 = np.asarray([6, 7, 8, 14, 28, 29, 30])
    col_set_eq_0 = np.asarray([25, 26, 27])

    # Combine the interesting columns.
    col_set_DER_PRI = np.sort(np.asarray([col_DER_mass_MMC, col_PRI_jet_num]))
    col_set_all = np.sort(np.append(np.append(col_set_DER_PRI, col_set_less_1), col_set_eq_0))
    col_set_DER_PRI_less_1 = np.sort(np.append(col_set_DER_PRI, col_set_less_1))

    # First split depending on the fact that DER_mass_MMC is defined or not.
    DER_MASS_MMC_undef = x[x[:, col_DER_mass_MMC] == UNDEFINED_VALUE]
    DER_MASS_MMC_def = x[x[:, col_DER_mass_MMC] != UNDEFINED_VALUE]

    # Second split depending on the four values of PRI_jet_num.
    DER_MASS_MMC_undef_PRI_jet_num_0 = DER_MASS_MMC_undef[DER_MASS_MMC_undef[:, col_PRI_jet_num] == 0]
    DER_MASS_MMC_undef_PRI_jet_num_1 = DER_MASS_MMC_undef[DER_MASS_MMC_undef[:, col_PRI_jet_num] == 1]
    DER_MASS_MMC_undef_PRI_jet_num_2 = DER_MASS_MMC_undef[DER_MASS_MMC_undef[:, col_PRI_jet_num] == 2]
    DER_MASS_MMC_undef_PRI_jet_num_3 = DER_MASS_MMC_undef[DER_MASS_MMC_undef[:, col_PRI_jet_num] == 3]

    DER_MASS_MMC_def_PRI_jet_num_0 = DER_MASS_MMC_def[DER_MASS_MMC_def[:, col_PRI_jet_num] == 0]
    DER_MASS_MMC_def_PRI_jet_num_1 = DER_MASS_MMC_def[DER_MASS_MMC_def[:, col_PRI_jet_num] == 1]
    DER_MASS_MMC_def_PRI_jet_num_2 = DER_MASS_MMC_def[DER_MASS_MMC_def[:, col_PRI_jet_num] == 2]
    DER_MASS_MMC_def_PRI_jet_num_3 = DER_MASS_MMC_def[DER_MASS_MMC_def[:, col_PRI_jet_num] == 3]

    # Remove all the columns that are now useless since they are the same for each set of data.
    DER_MASS_MMC_undef_PRI_jet_num_0 = np.delete(DER_MASS_MMC_undef_PRI_jet_num_0, col_set_all, axis=1)
    DER_MASS_MMC_undef_PRI_jet_num_1 = np.delete(DER_MASS_MMC_undef_PRI_jet_num_1, col_set_DER_PRI_less_1, axis=1)
    DER_MASS_MMC_undef_PRI_jet_num_2 = np.delete(DER_MASS_MMC_undef_PRI_jet_num_2, col_set_DER_PRI, axis=1)
    DER_MASS_MMC_undef_PRI_jet_num_3 = np.delete(DER_MASS_MMC_undef_PRI_jet_num_3, col_set_DER_PRI, axis=1)

    DER_MASS_MMC_def_PRI_jet_num_0 = np.delete(DER_MASS_MMC_def_PRI_jet_num_0, col_set_all, axis=1)
    DER_MASS_MMC_def_PRI_jet_num_1 = np.delete(DER_MASS_MMC_def_PRI_jet_num_1, col_set_DER_PRI_less_1, axis=1)
    DER_MASS_MMC_def_PRI_jet_num_2 = np.delete(DER_MASS_MMC_def_PRI_jet_num_2, col_set_DER_PRI, axis=1)
    DER_MASS_MMC_def_PRI_jet_num_3 = np.delete(DER_MASS_MMC_def_PRI_jet_num_3, col_set_DER_PRI, axis=1)

    ys = np.asarray([DER_MASS_MMC_undef_PRI_jet_num_0[:,1],
                     DER_MASS_MMC_undef_PRI_jet_num_1[:,1],
                     DER_MASS_MMC_undef_PRI_jet_num_2[:,1],
                     DER_MASS_MMC_undef_PRI_jet_num_3[:,1],
                     DER_MASS_MMC_def_PRI_jet_num_0[:,1],
                     DER_MASS_MMC_def_PRI_jet_num_1[:,1],
                     DER_MASS_MMC_def_PRI_jet_num_2[:,1],
                     DER_MASS_MMC_def_PRI_jet_num_3[:,1]])
    txs = np.asarray([DER_MASS_MMC_undef_PRI_jet_num_0[:,2:-1], # Last column is constant 0.
                      DER_MASS_MMC_undef_PRI_jet_num_1[:,2:],
                      DER_MASS_MMC_undef_PRI_jet_num_2[:,2:],
                      DER_MASS_MMC_undef_PRI_jet_num_3[:,2:],
                      DER_MASS_MMC_def_PRI_jet_num_0[:,2:-1], # Last column is constant 0.
                      DER_MASS_MMC_def_PRI_jet_num_1[:,2:],
                      DER_MASS_MMC_def_PRI_jet_num_2[:,2:],
                      DER_MASS_MMC_def_PRI_jet_num_3[:,2:]])
    ids = np.asarray([DER_MASS_MMC_undef_PRI_jet_num_0[:,0],
                      DER_MASS_MMC_undef_PRI_jet_num_1[:,0],
                      DER_MASS_MMC_undef_PRI_jet_num_2[:,0],
                      DER_MASS_MMC_undef_PRI_jet_num_3[:,0],
                      DER_MASS_MMC_def_PRI_jet_num_0[:,0],
                      DER_MASS_MMC_def_PRI_jet_num_1[:,0],
                      DER_MASS_MMC_def_PRI_jet_num_2[:,0],
                      DER_MASS_MMC_def_PRI_jet_num_3[:,0]])

    return ys, txs, ids

def standardize(x):
    centered = x - np.mean(x, axis=0)
    normed = centered / np.std(centered, axis=0)

    return normed

def build_model_data(y, x, ids):
    ys, txs, ids = split_data_meaningfuly(y, x, ids)
    txs = np.asarray(list(map(lambda tx: np.c_[np.ones(len(tx)), standardize(tx)], txs)))

    return ys, txs, ids

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))

        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx

    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)

        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
