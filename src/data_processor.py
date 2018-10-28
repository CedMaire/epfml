import numpy as np

"""
Functions related to data processing.
"""

# Value representing the fact that data is undefined or not relevant.
UNDEFINED_VALUE = -999

# All the different interesting columns.
col_DER_mass_MMC = 2
col_PRI_jet_num = 24
col_set_less_1 = np.asarray([6, 7, 8, 14, 28, 29, 30]) # All the columns undefined if PRI_jet_num <= 1
col_set_eq_0 = np.asarray([25, 26, 27]) # All the columns undefined if PRI_jet_num == 0

def split_data_meaningfuly(y, x, ids):
    """
    Splits the data into 8 subsets depending on the values in columns DER_mass_MMC and PRI_jet_num.
    DER_mass_MMC can be undefined without any reason, so we split the data into two set depending on the fact that this
        column is defined or not.
    PRI_jet_num can only contain 4 different values (0, 1, 2, 3): we split the two data subsets into 4 depending on these
        4 values.
    Once we have the 8 subsets we can begin cleaning the data by removing specific columns that will contain the exact
        same value throughout the whole set (DER_mass_MMC and PRI_jet_num) or that will be undefined because of the
        in PRI_jet_num (see the official PDF given by the Kaggle [http://higgsml.lal.in2p3.fr/files/2014/04/documentation_v1.8.pdf]).
    The returned triplet form 8 subsets of the input data, the 3 arrays are in the same order, i.e. ys[i], txs[i] and ids[i]
        form one specific subset.

    :param y: vector containing the expected labels
    :param x: data matrix (rows are samples, columns are features)
    :param ids: vector containing the ids of the samples
    :returns: ys - array of vectors of labels
              txs - array of matrices of the data features (rows are samples, columns are features)
              ids - array of vectors of ids
    """

    # First we add the column of ids and labels so that we get back the correct number of colums.
    x = np.c_[ids, y, x]

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

    # Remove all the columns that are now useless since they are the same for each set of data or undefined.
    DER_MASS_MMC_undef_PRI_jet_num_0 = np.delete(DER_MASS_MMC_undef_PRI_jet_num_0, col_set_all, axis=1)
    DER_MASS_MMC_undef_PRI_jet_num_1 = np.delete(DER_MASS_MMC_undef_PRI_jet_num_1, col_set_DER_PRI_less_1, axis=1)
    DER_MASS_MMC_undef_PRI_jet_num_2 = np.delete(DER_MASS_MMC_undef_PRI_jet_num_2, col_set_DER_PRI, axis=1)
    DER_MASS_MMC_undef_PRI_jet_num_3 = np.delete(DER_MASS_MMC_undef_PRI_jet_num_3, col_set_DER_PRI, axis=1)

    DER_MASS_MMC_def_PRI_jet_num_0 = np.delete(DER_MASS_MMC_def_PRI_jet_num_0, col_set_all, axis=1)
    DER_MASS_MMC_def_PRI_jet_num_1 = np.delete(DER_MASS_MMC_def_PRI_jet_num_1, col_set_DER_PRI_less_1, axis=1)
    DER_MASS_MMC_def_PRI_jet_num_2 = np.delete(DER_MASS_MMC_def_PRI_jet_num_2, col_set_DER_PRI, axis=1)
    DER_MASS_MMC_def_PRI_jet_num_3 = np.delete(DER_MASS_MMC_def_PRI_jet_num_3, col_set_DER_PRI, axis=1)

    # Creation of the output arrays.
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
    """
    Standardizes a data set. For each column it substract the mean of this feature and then devides each column by the
    standard variation of this column (everything is done element-wise).

    :param x: matrix of data
    :returns: the standardized matrix
    """

    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)

    x = x - x_mean
    return x / x_std

def build_model_data(y, x, ids):
    """
    Builds a model of data to be used for machine learning algorithms. Instead of creatung one model this method returns
    8 different models. A model is composed of a triplet: the expected labels (y), the matrix of samples and features (tx)
    and the id of the samples (ids). This method return 8 complete models, so ys is a list of expected labels, txs a list
    of matrices of samples and features and ids a list of id. yys, txs and ids all share the same ordering, this means that
    a same index forms one data model.

    :param y: vector of expected labels
    :param x: matrix of data (rows are samples, columns are features)
    :param ids: vector of sample ids
    :returns: ys - array of vectors of labels
              txs - array of matrices of the data features (rows are samples, columns are features)
              ids - array of vectors of ids
    """

    ys, txs, ids = split_data_meaningfuly(y, x, ids)
    txs = np.asarray(list(map(lambda tx: np.c_[np.ones(len(tx)), standardize(tx)], txs)))

    return ys, txs, ids
