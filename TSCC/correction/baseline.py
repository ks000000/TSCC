import numpy as np
import pandas as pd

def BASE_NA_byMode(df_fea, df_tar, config):
    # Raw value model filled with mode
    r"""
    Replace your None values with the most common number in your dataframe.

    Parameters
    ----------
    df_fea : pandas dataframe
        Your selected dataframe with None values.
    df_tar : None
        Not used yet.
    config : object
        Name of the column to be changed.

    Returns
    -------
    pandas series
        Your selected dataframe without None values.

    Examples
    --------
    >>> d = {'col1': [1, 9, None, 6, 20, None, 3, 1, 30, None]}
    >>> df = pd.DataFrame(data=d)
    >>> config = TSCC.preprocessing.Config(colname_raw = 'col1')
    >>> test = TSCC.correction.BASE_NA_byMode(df, 10,config)
    >>> test
    0     1.0
    1     9.0
    2     1.0
    3     6.0
    4    20.0
    5     1.0
    6     3.0
    7     1.0
    8    30.0
    9     1.0
    Name: col1, dtype: float64
    """

    return df_fea[config.colname_raw].fillna(df_fea[config.colname_raw].mode(dropna=True)[0]).fillna(0)

def BASE_NA_inSpace(df_fea, df_tar, config, cols_in_space = ["N", "E", "S", "W"]):

    r"""
    Replace your None values with the mean value of your columns cols_in_space, e.g. "N, E, S, W".

    Parameters
    ----------
    df_fea : pandas dataframe
        Your selected dataframe with columns "N, E, S, W". The names of these four columns HAVE to be there.
    df_tar : None
        Not used for now.
    config : object
        Name of the column to be changed.

    Returns
    -------
    pandas series
        Your selected dataframe without None values in the selected column.

    Examples
    --------
    >>> data = {
    >>>     "val_raw": [1, None, np.nan, 1, None],
    >>>     "isError": [False, True, True, False, True],
    >>>     'N': [2, 2, 2, 2, 2],
    >>>     'E': [3, 3, None, 3, 3],
    >>>     'S': [4, None, 4, 4, 4],
    >>>     'W': [None, 5, 5, 5, 5]
    >>> }
    >>> df_fea = pd.DataFrame(data)
    >>> config = TSCC.preprocessing.Config(colname_raw = 'val_raw')
    >>> TSCC.correction.BASE_NA_withSpecFeature(df_fea, None, config, "N")
    0    1.000000
    1    3.333333
    2    3.666667
    3    1.000000
    4    3.500000
    Name: val_raw, dtype: float64
    """

    # Raw value model filled with mode
    if all(col in df_fea.columns for col in cols_in_space):
        mean_space = df_fea[cols_in_space].mean(axis = 1)
        return df_fea[config.colname_raw].fillna(mean_space).fillna(0)
    else:
        #placeholder
        return pd.Series([np.nan]*df_fea.shape[0], index = df_fea.index)

def BASE_NA_withSpecFeature(df_fea, df_tar, config, feature):

    r"""
    Fill your missing values of a specific column using values form another column in your dataframe.

    Parameters
    ----------
    df_fea : pandas dataframe
        Your specific column with missing values.
    df_tar : None
        Not used for now.
    config : object
        Contains the name of your column with missing values.
    feature : str or list of str
        The name of the column to use for replacing the missing values.

    Returns
    -------
    pandas series
        Your new series with replaced values if the name of your feature column exists.
    Examples
    --------
    >>> data = {
    >>>     "val_raw": [1, None, np.nan, 1, None],
    >>>     "isError": [False, True, True, False, True],
    >>>     'N': [2, 2, 2, 2, 2],
    >>>     'E': [3, 3, None, 3, 3],
    >>>     'S': [4, None, 4, 4, 4],
    >>>     'W': [None, 5, 5, 5, 5]
    >>> }
    >>> df_fea = pd.DataFrame(data)
    >>> config = TSCC.preprocessing.Config(colname_raw = 'val_raw')
    >>> TSCC.correction.BASE_NA_withSpecFeature(df_fea, None, config, "N")
    0    1.0
    1    2.0
    2    2.0
    3    1.0
    4    2.0
    Name: val_raw, dtype: float64
    """

    # Raw value model filled with specific feature
    if all(col in df_fea.columns for col in [feature]):
        return df_fea[config.colname_raw].fillna(df_fea[feature]).fillna(0)
    else:
        #placeholder
        return pd.Series([np.nan]*df_fea.shape[0], index = df_fea.index)

def BASE_useDifferentFeature_NA_ByMode(df_fea, df_tar, config, feature):

    r"""
    Fills missing values (NaN) in a specified feature column using the most frequent value (mode).
    If the feature column does not exist, it returns a placeholder series filled with NaN.

    Parameters
    ------------
    df_fea : pandas dataframe
        The DataFrame containing the feature columns.
    df_tar : None
        Not used for now.
    config : object
        Contains the name of your column with missing values.
    feature : str or list of str
        The name (or list of names) of the column(s) to fill missing values in.

    Returns
    --------
    pandas series
        The column(s) from `df_fea` with missing values filled by mode and then 0.
        If the feature column does not exist, returns a series of NaN values with the same index as `df_fea`.

    Examples
    ----------
    >>> data = {
    >>>     "val_raw": [1, None, np.nan, 1, None],
    >>>     "isError": [False, True, True, False, True],
    >>>     'N': [2, 2, 2, 2, 2],
    >>>     'E': [3, 3, None, 3, 3],
    >>>     'S': [4, None, 4, 4, 4],
    >>>     'W': [None, 1, 2, 3, 4]
    >>> }
    >>> df_fea = pd.DataFrame(data)
    >>> config = TSCC.preprocessing.Config(colname_raw = 'val_raw')
    >>> TSCC.correction.BASE_useDifferentFeature_NA_ByMode(df_fea, None, config, "W")
    0    1.0
    1    1.0
    2    2.0
    3    3.0
    4    4.0
    Name: W, dtype: float64
    """
    if all(col in df_fea.columns for col in feature):
        return df_fea[feature].fillna(df_fea[feature].mode(dropna=True)[0]).fillna(0)
    else:
        #placeholder
        return pd.Series([np.nan]*df_fea.shape[0], index = df_fea.index)
