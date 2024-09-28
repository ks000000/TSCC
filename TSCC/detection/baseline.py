import numpy as np
import pandas as pd

def BASE_always_false_series(df_fea, df_tar, config):
    # Returns a series of True values with the same length as the input DataFrame.

    r"""
    Generates a Pandas Series filled with `False` values, matching the length of the input DataFrame.

    Parameters
    ----------
    df_fea : pandas dataframe
        The input DataFrame for which the series of `False` values is created.
    df_tar : None
        Not used yet.
    config : None
        Not used for now.

    Returns
    -------
    pandas series
        A Pandas Series of the same length as `df_fea`, filled entirely with `False` values.

    Examples
    --------
    >>> data = {
    >>>     "val_raw": [1, None, np.nan, 5, None],
    >>> }
    >>> df_fea = pd.DataFrame(data)
    >>> TSCC.detection.BASE_always_false_series(df_fea, None, None)
    0    False
    1    False
    2    False
    3    False
    4    False
    dtype: bool
    """

    return pd.Series([False] * df_fea.shape[0], index=df_fea.index)

def BASE_always_true_series(df_fea, df_tar, config):
    # Returns a series of True values with the same length as the input DataFrame.

    r"""
    Generates a Pandas Series filled with `True` values, matching the length of the input DataFrame.

    Parameters
    ----------
    df_fea : pandas dataframe
        The input dataframe for which the series of `True` values is created.
    df_tar : None
        Not used for now.
    config : None
        Not used yet.

    Returns
    -------
    pandas series
        A Pandas Series of the same length as `df_fea`, filled entirely with `True` values.

    Examples
    --------
    >>> data = {
    >>>     "val_raw": [1, None, np.nan, 5, None],
    >>> }
    >>> df_fea = pd.DataFrame(data)
    >>> TSCC.detection.BASE_always_true_series(df_fea, None, None)
    0    True
    1    True
    2    True
    3    True
    4    True
    dtype: bool
    """

    return pd.Series([True] * df_fea.shape[0], index=df_fea.index)

def BASE_det_perfect(df_fea, df_tar, config):

    r"""
    Compares two columns from two DataFrames and returns a boolean Series indicating if the values differ.

    Parameters
    ----------
    df_fea : pandas dataframe
        The DataFrame containing the feature data.
    df_tar : pandas dataframe
        The DataFrame containing the target or corrected data.
    config : object
        A configuration object containing:
        - colname_raw: the name of the column in `df_fea` to be compared.
        - colname_target_corr: the name of the column in `df_tar` to compare against.

    Returns
    -------
    pandas series
        A boolean Series where `True` indicates that the values in `df_fea[colname_raw]` and
        `df_tar[colname_target_corr]` are different, and `False` indicates they are the same.

    Examples
    --------
    >>> data_fea = {"val_raw": [1, 2, 3, 4, 5]
    >>> }
    >>> data_tar = {"val_tar": [1, 2, 0, 4, 0]
    >>> }
    >>> df_fea = pd.DataFrame(data_fea)
    >>> df_tar = pd.DataFrame(data_tar)
    >>> config = TSCC.preprocessing.Config(colname_raw='val_raw', colname_target_corr='val_tar')
    >>> TSCC.detection.BASE_det_perfect(df_fea, df_tar, config)
    0    False
    1    False
    2     True
    3    False
    4     True
    dtype: bool
    """

    series_fea = df_fea[config.colname_raw]
    series_tar = df_tar[config.colname_target_corr]
    # Returns a series of True values with the same length as the input DataFrame.
    return pd.Series(series_tar != series_fea, index=series_fea.index)
