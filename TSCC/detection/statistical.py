import pandas as pd
import numpy as np
from scipy import signal
from statsmodels.tsa.seasonal import seasonal_decompose

def STAT_byDistFromCenter(series, eps, center_measure='median', dynamic_window=None):
    r"""

    Check if observation is within reasonable distance to center,
    eps and eps_pro define the distance.

    Parameters
    ----------
    series : pandas series
    eps : float
        Define a threshold (epsilon) for the acceptable distance (only positive value > 0) from the center
    center_measure : string, optional
        choose from list ['median', 'mean']
    dynamic_window : int, optional
        Epsilon Pro version, epsilon threshold is dynamic according to the window frame

    Returns
    -------
    s : pandas series
        float values, TRUE (1.0) if value is regarded as correct by this method FALSE (0.0) otherwise

    Examples
    --------
    >>> time_index = pd.date_range(start='2023-01-01 00:00', periods=7, freq='30T')
    >>> s = pd.Series([np.nan, 1, 2.0, "string", 5.4, -999,  None], index = time_index)
    >>> TSCC.detection.STAT_byDistFromCenter(s, 3)
    2023-01-01 00:00:00    0.0
    2023-01-01 00:30:00    0.0
    2023-01-01 01:00:00    0.0
    2023-01-01 01:30:00    0.0
    2023-01-01 02:00:00    0.5
    2023-01-01 02:30:00    0.5
    2023-01-01 03:00:00    0.0
    Freq: 30T, dtype: float64
    """
    # Convert non-numeric values to NaN
    series = pd.to_numeric(series, errors='coerce')

    if center_measure == 'median':
        center = series.median()
    elif center_measure == 'mean':
        center = series.mean()
    else:
        return pd.Series()

    if dynamic_window:
        if center_measure == 'median':
            rolling_center = series.rolling(dynamic_window, min_periods=1).median()
        elif center_measure == 'mean':
            rolling_center = series.rolling(dynamic_window, min_periods=1).mean()
        else:
            return pd.Series()
        # here: using 2 standard deviations for calculating eps_pro
        eps = rolling_center.std() * 2

    error_index = abs(series - center) > eps

    # no error, but observations doubtful
    return (error_index).astype(float).replace(1.0, 0.5)


def STAT_byDistFromCenterRolling(series, threshold, window=3):
    r"""
    Check if values are within static distance to rolling center.

    Parameters
    ----------
    series : pandas series
    threshold : number, optional #nicht ganz sicher ob wirklich optional
        The default is 1.
    window : int, optional
        Rolling center according to the window frame
    center : boolean, optional
        The default is False.

    Returns
    -------
    s : pandas series
        boolean values, TRUE if value is regarded as correct by this method FALSE otherwise


    Examples
    --------
    >>> time_index = pd.date_range(start='2023-01-01 00:00', periods=7, freq='30T')
    >>> s = pd.Series([np.nan, 1, 2.0, "string", 5.4, -999,  None], index = time_index)
    >>> TSCC.detection.STAT_byDistFromCenterRolling(s, 3, 2)
    2023-01-01 00:00:00    0.0
    2023-01-01 00:30:00    0.0
    2023-01-01 01:00:00    0.0
    2023-01-01 01:30:00    0.0
    2023-01-01 02:00:00    0.0
    2023-01-01 02:30:00    0.5
    2023-01-01 03:00:00    0.0
    Freq: 30T, dtype: float64
    """
    # Convert non-numeric values to NaN
    series = pd.to_numeric(series, errors='coerce')

    rolling = series.rolling(window, min_periods=1).mean()

    return (abs(rolling - series) > threshold).astype(float).replace(1.0, 0.5)

def STAT_byIQR(series, lo=0.25, up=0.75, k=1.5):
    r"""
    Check if values are in the scaled interquantile range.

    Parameters
    ----------
    series : pandas series
    lo : number, optional
        Percentage value in range [0, 1]. The default is 0.25.
    up : number, optional
        Percentage value in range [0, 1]. The default is 0.75.
    k : number, optional
        Scaling factor of regular IQR to determine outliers,
        typical scaling factor 1.5 as default

    Returns
    -------
    s : pandas series
        boolean values, TRUE if value is regarded as correct by this method FALSE otherwise

    Examples
    --------
    >>> time_index = pd.date_range(start='2023-01-01 00:00', periods=7, freq='30T')
    >>> s = pd.Series([np.nan, 1, 2.0, "string", 5.4, -999,  None], index = time_index)
    >>> TSCC.detection.STAT_byIQR(s)
    2023-01-01 00:00:00    0.5
    2023-01-01 00:30:00    0.0
    2023-01-01 01:00:00    0.0
    2023-01-01 01:30:00    0.5
    2023-01-01 02:00:00    0.0
    2023-01-01 02:30:00    0.5
    2023-01-01 03:00:00    0.5
    Freq: 30T, dtype: float64
    """
    # Convert non-numeric values to NaN
    series = pd.to_numeric(series, errors='coerce')

    # lo_val, up_val = IQRBoundary(series, lo, up, k)
    lo, up = series.quantile([lo, up]).values
    iqr = (up - lo)
    lo_th, up_th = [lo -  (k * iqr), up +  (k * iqr)]

    # check if series values in boundary
    isValid = (lo_th < series) & (series < up_th)
    isDoubtful = ~isValid
    return isDoubtful.astype(float).replace(1.0, 0.5)

def STAT_byIQRdiff(series, lo=0.25, up=0.75, k=1.5):
    r"""
    Check if differences to previous values are in the scaled interquantile range.

    Parameters
    ----------
    series : pandas series
    lo : number, optional
        Percentage value in range [0, 1]. The default is 0.25.
    up : number, optional
        Percentage value in range [0, 1]. The default is 0.75.
    k : number, optional
        Scaling factor of regular IQR to determine outliers,
        typical scaling factor 1.5 as default

    Returns
    -------
    s : pandas series
        boolean values, TRUE if value is regarded as correct by this method FALSE otherwise

    Examples
    --------
    >>> time_index = pd.date_range(start='2023-01-01 00:00', periods=7, freq='30T')
    >>> s = pd.Series([np.nan, 1, 2.0, "string", 5.4, -999,  None], index = time_index)
    >>> TSCC.detection.STAT_byIQRdiff(s)
    2023-01-01 00:00:00    0.0
    2023-01-01 00:30:00    0.5
    2023-01-01 01:00:00    0.0
    2023-01-01 01:30:00    0.5
    2023-01-01 02:00:00    0.5
    2023-01-01 02:30:00    0.0
    2023-01-01 03:00:00    0.5
    Freq: 30T, dtype: float64
    """

    # Convert non-numeric values to NaN
    series = pd.to_numeric(series, errors='coerce')

    # difference to previous value
    series_diff = series.diff()

    #lo, up = IQRBoundary(series_diff, lo, up, k)
    lo, up = series_diff.quantile([lo, up]).values
    iqr = k * (up - lo)
    lo, up = [lo - iqr, up + iqr]

    isValid = (lo < series_diff) & (series_diff < up)
    # first entry is NaN (no previous value to diff from) so it is set as true
    isValid[:1] = True
    isDoubtful = ~isValid
    return isDoubtful.astype(float).replace(1.0, 0.5)


def STAT_byZScore(series, z = 3, b_modified = False, k=0.6745):
    r"""
    Check if values are within a standardized score from the mean.
    Best for normally distributed data.

    Parameters
    ----------
    series : pandas series
    z : float, optional
        z-score for outliers.
        Typically, values are considered as outliers when abs(z)>3.
    b_modified : boolean, optional
        Weather modified z-score is used or not.
    k : float, optional
        The default is 0.6745.

    Returns
    -------
    pandas series
        Series of index of outliers or cleaned data of the original series.

    Examples
    --------
    >>> time_index = pd.date_range(start='2023-01-01 00:00', periods=7, freq='30T')
    >>> s = pd.Series([np.nan, 1, 2.0, "string", 5.4, -999,  None], index = time_index)
    >>> TSCC.detection.STAT_byZScore(s)
    2023-01-01 00:00:00    0.0
    2023-01-01 00:30:00    0.0
    2023-01-01 01:00:00    0.0
    2023-01-01 01:30:00    0.0
    2023-01-01 02:00:00    0.0
    2023-01-01 02:30:00    0.0
    2023-01-01 03:00:00    0.0
    Freq: 30T, dtype: float64
    """

    def zScore(series, b_modified=False, k=0.6745):

        r"""
        Modified method according to Boris Iglewicz and David Hoaglin (1993),
        "Volume 16: How to Detect and Handle Outliers", The ASQC Basic References in Quality Control: Statistical
        Techniques, Edward F. Mykytka, Ph.D., Editor.

        Parameters
        ----------
        series : pandas dataframe
        b_modified : boolean, optional
            Wether modified z score is used or not.
            The default is False.
        k : int, optional
            The default is 0.6745.

        Returns
        -------
        b_series : pandas series
            z-score for the series.

        Examples
        --------
        >>> dates = np.arange('2021-07-24 00:00:00', '2021-07-24 10:00:00', dtype='datetime64[h]')
        >>> d = {'col1': [1, 9, 3, 6, 20, 8, 3, 1, 30, 5]}
        >>> df_time = pd.DataFrame(data=d, index=dates)
        >>> test = zScore(series=df_time['col1'], b_modified=False, k=0.6745)
        2021-07-24 00:00:00   -0.813042
        2021-07-24 01:00:00    0.042792
        2021-07-24 02:00:00   -0.599084
        2021-07-24 03:00:00   -0.278146
        2021-07-24 04:00:00    1.219563
        2021-07-24 05:00:00   -0.064188
        2021-07-24 06:00:00   -0.599084
        2021-07-24 07:00:00   -0.813042
        2021-07-24 08:00:00    2.289356
        2021-07-24 09:00:00   -0.385125
        Name: col1, dtype: float64
        """

        if not b_modified:
            b_series = (series - series.mean()) / series.std()
        else:
            b_series = series - series.median()
            b_series = k * b_series / abs(b_series).median()
        return b_series  # ,np.quantile(series, 0.95)

    # Convert non-numeric values to NaN
    series = pd.to_numeric(series, errors='coerce')

    b_series = abs( zScore(series, b_modified, k=k)) > z
    return b_series.astype(float).replace(1.0, 0.5)


def STAT_byCorrelationAnalysis(df, target_col, correlation_threshold=0.2):

    r"""
    Check the consistency of the target variable (e.g., rainfall) by comparing it with other variables.

    Parameters
    ----------
    df : pandas dataframe
        The dataframe contains the target variable and other meteorological variables.
    target_col : str
        The target variable (e.g., 'rainfall') to be validated.
    correlation_threshold : float
        Minimum acceptable correlation between the target variable and other variables.

    Returns
    -------
    pandas series
        A series with an additional column 'consistency_error' (True if error, False otherwise).

    Examples
    --------
    >>> nr_obs = 6
    >>> np.random.seed(0)
    >>> # Generate random data
    >>> data = np.random.randn(nr_obs, 4)
    >>> time_index = pd.date_range(start='2023-01-01 00:00', periods=nr_obs, freq='30T')
    >>> # Create DataFrame with specified column names
    >>> df = pd.DataFrame(data, columns=["ground_truth", "raw", 'fea_1', 'fea_2'], index = time_index)
    >>> df["raw"] = df["ground_truth"] + np.random.normal(0, 5, nr_obs)*np.random.randint(0, 2, nr_obs)
    >>> df["isCorrect_gt"] = df["ground_truth"] == df["raw"]
    >>> TSCC.detection.STAT_byCorrelationAnalysis(df, "raw")
    2023-01-01 00:00:00    0.5
    2023-01-01 00:30:00    0.0
    2023-01-01 01:00:00    0.0
    2023-01-01 01:30:00    0.0
    2023-01-01 02:00:00    0.0
    2023-01-01 02:30:00    0.5
    Freq: 30T, Name: consistency_error, dtype: float64
    """

    # Calculate correlations between the target variable and other variables
    correlations = df.corr()[target_col].drop(target_col)

    # Identify variables with low correlation to the target variable
    inconsistent_vars = correlations[abs(correlations) < correlation_threshold].index.tolist()

    # Check for inconsistency by comparing with these low correlation variables
    df['consistency_error'] = False

    for var in inconsistent_vars:
        # Flag as inconsistent if the target variable significantly deviates from expected behavior
        # Here, we assume that a deviation larger than a standard deviation is suspicious
        target_mean = df[target_col].mean()
        target_std = df[target_col].std()
        deviation = abs(df[target_col] - df[var])

        df['consistency_error'] |= deviation > target_std

    # no error, but observations doubtful
    return df['consistency_error'].astype(float).replace(1.0, 0.5)



