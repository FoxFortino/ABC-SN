import numpy as np
import pandas as pd
from keras.utils import to_categorical


def extract_dataframe(sn_data):
    """
    Extract both metadata and flux data from a dataframe.
    """
    # Extract the row indices from the dataframe. These correspond to the SN
    # name of the spectrum at each row.
    index = sn_data.index

    # Extract the sub-dataframe that contains only the columns corresponding
    # to flux values. We do this specifically with a regex expression that
    # takes only the columns that start with a number.
    df_fluxes = sn_data.filter(regex="\d+")
    fluxes = df_fluxes.to_numpy(dtype=float)

    # Extract the columns that identify the flux columns. These will also be
    # the wavelengths at for each flux value, but as a string.
    flux_columns = df_fluxes.columns
    wvl = flux_columns.to_numpy(dtype=float)

    # In the even that more non-flux columns are added to these dataframes, we
    # find all of the columns representing the metadata (such as SN class,
    # spectral phase, etc.) by extracting all columns apart from
    # `flux_columns`.
    metadata_columns = sn_data.columns.difference(flux_columns)
    df_metadata = sn_data[metadata_columns]

    return (index, wvl,
            flux_columns, metadata_columns,
            df_fluxes, df_metadata,
            fluxes)


def extract(df, return_wvl=False):
    data = extract_dataframe(df)
    X = data[6]
    Y = data[5]["SN Subtype ID"].to_numpy(dtype=int)
    
    N = X.shape[0]
    num_wvl = X.shape[1]
    num_classes = np.unique(Y).size

    # If you have removed some SN classes from the dataset, then you will find that the class IDs in Y will not work with the Keras `to_categorical` function. For example, if you remove the SN Ia-csm class from the dataset, when you calculate `np.unique(Y)` you will find it goes `[0, 1, 2, 4, ...]` because Ia-csm corresponded to class ID `3`. So the following block of code will create a dictionary that transformers the class IDs into new values that are consecutive. This dictionary must later be used when constructing confusion matrices so it is returned.
    unique_Y = np.unique(Y)
    new_sn_ids = np.arange(np.unique(Y).size)
    sn_dict = {old_id: new_id for old_id, new_id in zip(unique_Y, new_sn_ids)}
    if num_classes != np.max(Y) + 1:
        Y = np.array([sn_dict[y] for y in Y])
    
    Y_OH = to_categorical(Y, num_classes=num_classes)

    if return_wvl:
        return X, Y_OH, N, num_wvl, num_classes, sn_dict, data[1]
    else:
        return X, Y_OH, N, num_wvl, num_classes, sn_dict


def add_dim(X, swap=False):
    X = X[..., None]
    if swap:
        X = np.swapaxes(X, 1, 2)
    return X

