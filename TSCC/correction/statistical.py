import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pykalman import KalmanFilter

def STAT_byFeature(df_fea, df_tar, df_flag_erroneous, config, feature):
    # retain the value from config.colname_raw if df_flag_erroneous is FALSE, or set it to np.nan if isValid_pred is False

    r"""
    Processes and fills values in the dataframe based on a flag indicating erroneous data and a specified feature column.
    It retains the values from the `config.colname_raw` column if the data is not erroneous
    and fills missing values from a feature column if necessary.

    Parameters
    ----------
    df_fea : pandas dataframe
        The DataFrame containing feature data.
    df_tar : None
        Not used for now.
    df_flag_erroneous : pandas series or dataframe
        A boolean series or DataFrame indicating whether each value is erroneous (`True`) or valid (`False`).
    config : object
        A configuration object that contains the column name `colname_raw` to process.
    feature : str
        The name of the column to use for filling missing values if `NaN` values exist in `config.colname_raw`.

    Returns
    -------
    pandas series
        A Series where missing values from `config.colname_raw` are first filled with values from the `feature` column,
        and then any remaining `NaN` values are filled with 0. If the `feature` column does not exist,
        a series of `NaN` values is returned.

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
    >>> TSCC.correction.STAT_byFeature(df_fea, None, "isError", config, "N")
    0    1.0
    1    2.0
    2    2.0
    3    1.0
    4    2.0
    Name: val_raw_pred, dtype: float64
    """

    df_fea[f"{config.colname_raw}_pred"] = np.where(~df_fea[df_flag_erroneous], df_fea[config.colname_raw], np.nan)
    if feature in df_fea.columns:
        return df_fea[f"{config.colname_raw}_pred"].fillna(df_fea[feature]).fillna(0)
    else:
        #placeholder
        return pd.Series([np.nan]*df_fea.shape[0], index = df_fea.index)


def STAT_byFilling(df_fea, df_tar, df_flag_erroneous, config, filler ):

    r"""
    Imputes missing values in the dataframe using a specified filling method.

    Parameters
    ----------
    df_fea : pandas dataframe
        The DataFrame containing feature data.
    df_tar : None
        Not used yet.
    df_flag_erroneous : pandas series or pandas dataframe
        A boolean series or DataFrame indicating whether each value is erroneous (`True`) or valid (`False`).
    config : object
        A configuration object containing the name of the column `colname_raw` to be processed.
    filler : str or float or int
        Specifies how to fill the missing values. Three options:
        - 'ffill': Propagates the last valid observation forward to the next valid.
        - 'backfill' or 'bfill': Uses the next valid observation to fill the gap.
        - <float or int>: Uses a custom numeric value to fill missing values.

    Returns
    -------
    pandas series
        A Series corresponding to the `{config.colname_raw}_pred` column with missing values filled based on the `filler` method.
        If `filler` is numeric, fills NaNs with that number. If `filler` is a method like 'ffill' or 'bfill', applies the method.

    Examples
    --------
    >>> data = {
    >>>     "val_raw": [1, None, np.nan, 8, None],
    >>>     "isError": [False, True, True, False, True],
    >>>     'N': [2, 2, 2, 2, 2],
    >>>     'E': [3, 3, None, 3, 3],
    >>>     'S': [4, None, 4, 4, 4],
    >>>     'W': [None, 5, 5, 5, 5]
    >>> }
    >>> df_fea = pd.DataFrame(data)
    >>> config = TSCC.preprocessing.Config(colname_raw = 'val_raw')
    >>> TSCC.correction.STAT_byFilling(df_fea, None, "isError", config, filler = "ffill")
    0    1.0
    1    1.0
    2    1.0
    3    8.0
    4    8.0
    Name: val_raw_pred, dtype: float64
    """

    df_fea[f"{config.colname_raw}_pred"] = np.where(~df_fea[df_flag_erroneous], df_fea[config.colname_raw], np.nan)
    return  df_fea[f"{config.colname_raw}_pred"].fillna( filler ) if type(filler) in [int, float] else df_fea.fillna( method = filler )[f"{config.colname_raw}_pred"]

def STAT_byMode(df_fea, df_tar, df_flag_erroneous, config):
    # retain the value from config.colname_raw if isValid_pred is True, or set it to np.nan if isValid_pred is False

    r"""
    Imputes missing values in the dataframe using the mode of the specified column.

    Parameters
    ----------
    df_fea : pandas dataframe
        The DataFrame containing feature data.
    df_tar : None
        Not used yet.
    df_flag_erroneous : pandas series or pandas dataframe
        A boolean series or DataFrame indicating whether each value is erroneous (`True`) or valid (`False`).
    config : object
        A configuration object containing the name of the column `colname_raw` to be processed.

    Returns
    -------
    pandas series
        A Series corresponding to the `{config.colname_raw}_pred` column with missing values filled based
        on the mode of `config.colname_raw`. If the mode is not available, remaining missing values are filled with 0.

    Examples
    --------
    >>> data = {
    >>>     "val_raw": [1, None, np.nan, 8, None],
    >>>     "isError": [False, True, True, False, True],
    >>>     'N': [2, 2, 2, 2, 2],
    >>>     'E': [3, 3, None, 3, 3],
    >>>     'S': [4, None, 4, 4, 4],
    >>>     'W': [None, 5, 5, 5, 5]
    >>> }
    >>> df_fea = pd.DataFrame(data)
    >>> config = TSCC.preprocessing.Config(colname_raw = 'val_raw')
    >>> TSCC.correction.STAT_byMode(df_fea, None, "isError", config)
    0    1.0
    1    1.0
    2    1.0
    3    8.0
    4    1.0
    Name: val_raw_pred, dtype: float64
    """

    df_fea[f"{config.colname_raw}_pred"] = np.where(~df_fea[df_flag_erroneous], df_fea[config.colname_raw], np.nan)
    return df_fea[f"{config.colname_raw}_pred"].fillna(df_fea[config.colname_raw].mode(dropna=True)[0]).fillna(0)

def STAT_inSpace(df_fea, df_tar, df_flag_erroneous, config, cols_in_space = ["N", "E", "S", "W"]):
    # retain the value from config.colname_raw if isValid_pred is True, or set it to np.nan if isValid_pred is False

    r"""
    Imputes missing values in the dataframe using the mean of specified spatial columns.

    Parameters
    ----------
    df_fea : pandas dataframe
        The DataFrame containing feature data.
    df_tar : None
        Not used for now.
    df_flag_erroneous : pandas series or pandas dataframe
        A boolean series or DataFrame indicating whether each value is erroneous (`True`) or valid (`False`).
    config : object
        A configuration object containing the name of the column `colname_raw` to be processed.
    cols_in_space : list of str, optional
        A list of column names representing spatial data (e.g., ["N", "E", "S", "W"]). Default is ["N", "E", "S", "W"].

    Returns
    -------
    pd.Series
        A Series corresponding to the `{config.colname_raw}_pred` column with missing values filled based on the mean
        of the specified spatial columns.If spatial columns are missing, the function returns a Series of `NaN` values.

    Examples
    --------
    >>> data = {
    >>>     "val_raw": [1, None, np.nan, 8, None],
    >>>     "isError": [False, True, True, False, True],
    >>>     'N': [2, 2, 2, 2, 2],
    >>>     'E': [3, 3, None, 3, 3],
    >>>     'S': [4, None, 4, 4, 4],
    >>>     'W': [None, 5, 5, 5, 5]
    >>> }
    >>> df_fea = pd.DataFrame(data)
    >>> config = TSCC.preprocessing.Config(colname_raw = 'val_raw')
    >>> TSCC.correction.STAT_inSpace(df_fea, None, "isError", config)
    0    1.000000
    1    3.333333
    2    3.666667
    3    8.000000
    4    3.500000
    Name: val_raw_pred, dtype: float64
    """

    df_fea[f"{config.colname_raw}_pred"] = np.where(~df_fea[df_flag_erroneous].astype(bool), df_fea[config.colname_raw], np.nan)
    if all(col in df_fea.columns for col in cols_in_space):
        mean_space = df_fea[cols_in_space].mean(axis = 1)
        return df_fea[f"{config.colname_raw}_pred"].fillna(mean_space).fillna(0)
    else:
        #placeholder
        return pd.Series([np.nan]*df_fea.shape[0], index = df_fea.index)


def STAT_byRollingMean(df_fea, df_tar, df_flag_erroneous, config, window_size=10):

    r"""
    Imputes missing values using the rolling mean of the specified window size from previous values.
    The returned series may still have missing values if there is no data available in the window.

    Parameters
    ----------
    df_fea : pandas dataframe
        The DataFrame containing feature data.
    df_tar : None
        Not used for now.
    df_flag_erroneous : pandas series or pandas dataframe
        A boolean series or DataFrame indicating whether each value is erroneous (`True`) or valid (`False`).
    config : object
        A configuration object containing the name of the column `colname_raw` to be processed.
    window_size : int, optional
        The size of the rolling window used to compute the mean. Default is 10.

    Returns
    -------
    pandas series
        A Series corresponding to the `{config.colname_raw}_pred` column with missing values filled using the rolling mean.
        The Series may still contain missing values if there is insufficient data within the window.

    Examples
    --------
    >>> data = {
    >>>     "val_raw": [1, None, np.nan, 8, None],
    >>>     "isError": [False, True, True, False, True],
    >>>     'N': [2, 2, 2, 2, 2],
    >>>     'E': [3, 3, None, 3, 3],
    >>>     'S': [4, None, 4, 4, 4],
    >>>     'W': [None, 5, 5, 5, 5]
    >>> }
    >>> df_fea = pd.DataFrame(data)
    >>> config = TSCC.preprocessing.Config(colname_raw = 'val_raw')
    >>> TSCC.correction.STAT_byRollingMean(df_fea, None, "isError", config)
    0    1.0
    1    1.0
    2    1.0
    3    8.0
    4    4.5
    Name: val_raw_pred, dtype: float64
    """

    df_fea[f"{config.colname_raw}_pred"] = np.where(~df_fea[df_flag_erroneous], df_fea[config.colname_raw], np.nan)

    series = df_fea[f"{config.colname_raw}_pred"]

    help_df = pd.DataFrame(series)
    help_df.columns = ["series"]
    # min_periods=1 needed to compute mean if at least one value exists
    help_df["rollingMean"] = series.rolling(window_size, min_periods=1).mean()
    help_df[f"{config.colname_raw}_pred"] = help_df["series"].fillna(help_df["rollingMean"])

    return help_df[f"{config.colname_raw}_pred"]

def STAT_byInterpolation(df_fea, df_tar, df_flag_erroneous, config, method='linear'):

    r"""
    Imputes missing values in the dataframe using interpolation.

    Parameters
    ----------
    df_fea : pandas dataframe
        The DataFrame containing feature data.
    df_tar : None
        Not used for now.
    df_flag_erroneous : pandas series or pandas dataframe
        A boolean series or DataFrame indicating whether each value is erroneous (`True`) or valid (`False`).
    config : object
        A configuration object containing the name of the column `colname_raw` to be processed.
    method : str, optional
        The interpolation method to use.
        Default is 'linear'.


    Returns
    -------
    pandas series
        A Series corresponding to the `{config.colname_raw}_pred` column with missing values imputed
        using the specified interpolation method.

    Examples
    --------
    >>> data = {
    >>>     "val_raw": [1, None, np.nan, 8, None],
    >>>     "isError": [False, True, True, False, True],
    >>>     'N': [2, 2, 2, 2, 2],
    >>>     'E': [3, 3, None, 3, 3],
    >>>     'S': [4, None, 4, 4, 4],
    >>>     'W': [None, 5, 5, 5, 5]
    >>> }
    >>> df_fea = pd.DataFrame(data)
    >>> config = TSCC.preprocessing.Config(colname_raw = 'val_raw')
    >>> TSCC.correction.STAT_byInterpolation(df_fea, None, "isError", config)
    0    1.000000
    1    3.333333
    2    5.666667
    3    8.000000
    4    4.500000
    Name: val_raw_pred, dtype: float64
    """

    df_fea[f"{config.colname_raw}_pred"] = np.where(~df_fea[df_flag_erroneous], df_fea[config.colname_raw], np.nan)

    series = df_fea[f"{config.colname_raw}_pred"]

    median = series.median()
    # if the first or last value is missing interpolate can not calculate the following respectively  privious values
    series.values[0] = median if np.isnan(series.values[0]) else series.values[0]
    series.values[-1] = median if np.isnan(series.values[-1]) else series.values[-1]
    return series.interpolate( method = method)

class KalmanFilter:
    def __init__(self, process_variance, measurement_variance, estimation_error, initial_estimate):
        """
        Initializes the Kalman filter.

        Args:
            process_variance: Variance in the process (how much the process changes over time).
            measurement_variance: Variance in the measurement (measurement noise).
            estimation_error: Initial estimation error (uncertainty in the initial estimate).
            initial_estimate: Initial guess of the state.
        """
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimation_error = estimation_error
        self.estimate = initial_estimate
        self.kalman_gain = 0

    def update(self, measurement):
        """
        Updates the estimate with a new measurement.

        Args:
            measurement: New measurement value.
        """
        # Prediction step: Update the estimation error based on the process variance
        self.estimation_error += self.process_variance

        # Kalman Gain calculation: How much to adjust based on measurement uncertainty
        self.kalman_gain = self.estimation_error / (self.estimation_error + self.measurement_variance)

        # Correction step: Update the estimate with the measurement
        self.estimate = self.estimate + self.kalman_gain * (measurement - self.estimate)

        # Update the estimation error after the measurement update
        self.estimation_error = (1 - self.kalman_gain) * self.estimation_error

        return self.estimate


def STAT_byKalman(df_fea, df_tar, df_flag_erroneous, config, process_variance=1e-5, measurement_variance=0.1):

    r"""
    Applies a Kalman Filter to noisy data to correct errors and smooth the series.

    Parameters
    ----------
    df_fea : pandas dataframe
        The DataFrame containing the feature data.
    df_tar : None
        Not used for now.
    df_flag_erroneous : pandas series or pandas dataframe
        A boolean series or DataFrame indicating whether each value is erroneous (`True`) or valid (`False`).
    config : object
        A configuration object that contains the name of the column to be processed (colname_raw).
    process_variance : float, optional
        The variance of the process noise. It controls how much the system dynamics are expected to change over time.
        Default is 1e-5.
    measurement_variance : float, optional
        The variance of the measurement noise. It indicates how noisy the measurements are.
        Default is 0.1.

    Returns
    -------
    pandas series
        A Series with missing or erroneous values replaced by Kalman Filtered values.

    Examples
    --------
    >>> data = {
    >>>     "val_raw": [1, 3, 4, 8, None],
    >>>     "isError": [False, True, True, False, True],
    >>>     'N': [2, 2, 2, 2, 2],
    >>>     'E': [3, 3, None, 3, 3],
    >>>     'S': [4, None, 4, 4, 4],
    >>>     'W': [None, 5, 5, 5, 5]
    >>> }
    >>> df_fea = pd.DataFrame(data)
    >>> config = TSCC.preprocessing.Config(colname_raw = 'val_raw')
    >>> TSCC.correction.STAT_byKalman(df_fea, None, "isError", config)
    0    1.000000
    1    1.952436
    2    2.613061
    3    8.000000
    4         NaN
    Name: val_raw_pred, dtype: float64
    """

    noisy_data = df_fea[config.colname_raw]

    # Initial guesses
    initial_estimate = noisy_data[0]  # Start with the first data point
    estimation_error = 1.0  # Initial estimation error

    # Create Kalman Filter object
    kf = KalmanFilter(process_variance, measurement_variance, estimation_error, initial_estimate)

    # Apply the filter to each measurement
    df_fea["kalman_filtered"] = [kf.update(measurement) for measurement in df_fea[config.colname_raw]]

    df_fea[f"{config.colname_raw}_pred"] = np.where(~df_fea[df_flag_erroneous], df_fea[config.colname_raw], np.nan)

    return df_fea[f"{config.colname_raw}_pred"].fillna(df_fea["kalman_filtered"])

