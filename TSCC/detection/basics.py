import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def BASIC_byExistance(series, no_data_char = None, obs_type = None):
    
    r"""
    Check if observation values exist and have the correct data type.

    Parameters
    ----------
    series : pandas series
        The input series to check for validity.
    no_data_char : optional
        A placeholder value for missing data.
    obs_type : data type, e.g. int, float, str

    Returns
    -------
    pandas series
        A series where 1.0 indicates an error (i.e., invalid observation) and 0.0 indicates no error.

    Examples
    --------
    >>> s = pd.Series([np.nan, 1, 2.0, "string", 5.4, -999,  None])
    >>> TSCC.detection.BASIC_byExistance(s, -999, float)
    0    1.0
    1    1.0
    2    0.0
    3    1.0
    4    0.0
    5    1.0
    6    1.0
    dtype: float64
    """
    is_data = pd.Series([True]*len(series), index = series.index)
    if no_data_char:
        is_data = (series != no_data_char)
    is_correct_dtype = pd.Series([True] * len(series), index=series.index)
    if obs_type:
        is_correct_dtype = series.apply(lambda x: isinstance(x, obs_type))

    isValid_bool = (~series.isna() & is_data & is_correct_dtype)
    isError = ~isValid_bool
    return isError.astype(float)

def BASIC_byRange(series, lower=None, upper=None):
    r"""
    Check if values are within plausible limits.

    Parameters
    ----------
    df_fea : pandas dataframe
        series of detMLConfig.colname_raw is used only
    df_tar : pandas dataframe
        only serves as placeholder, no usage of it
    upper : number, optional
        Upper boundary of the series. The default is None.
    lower : number, optional
        Lower boundary of the series. The default is None.

    Returns
    -------
    s : pandas series
        cleaned series.

    Examples
    --------
    The first example returns False for every value out of boundary.

    >>> test_boolean = STAT_byBoundary(df['col1'], upper=9, lower=1, setValueTo = "Boolean")
    0     True
    1     True
    2     True
    3     True
    4    False
    5     True
    6     True
    7     True
    8    False
    9     True
    Name: col1, dtype: bool


    The second example returns numpy.NaN for every value out of boundary, else the initial value.

    >>> BASIC_isValid_byRange(df['col1'], upper=9, lower=1, setValueTo = "OutOfBoundary_toNaN")
    0    1.0
    1    9.0
    2    3.0
    3    6.0
    4    NaN
    5    8.0
    6    3.0
    7    1.0
    8    NaN
    9    5.0
    Name: col1, dtype: float64


    The third example deletes every value which is out of boundary.

    >>> BASIC_isValid_byRange(df['col1'], upper=9, lower=1, setValueTo = "OutOfBoundary_Delete")
    0    1
    1    9
    2    3
    3    6
    5    8
    6    3
    7    1
    9    5
    Name: col1, dtype: int64
    """

    # Convert non-numeric values to NaN
    series = pd.to_numeric(series, errors='coerce')

    df = pd.DataFrame(series)

    if upper is not None or lower is not None:
        if upper is not None and lower is not None:
            df["exceedsRange"] = (df[series.name] > upper) | (df[series.name] < lower)
        elif upper is not None:
            df["exceedsRange"] = df[series.name] > upper
        else:
            df["exceedsRange"] = df[series.name] < lower
    else:
        df["exceedsRange"] = False
        print('No upper and lower boundary as input parameter - no changes made.')

    return df["exceedsRange"].astype(float)

def BASIC_byBoundaryVariance_60min(series, min_var):
    r"""
    Check if variance is too low.

    Parameters
    ----------
    series : panda series
        timestamp index
    min_var : string or integer or float
        minimal variability

    Returns
    -------
    boolean
            A series where `0.5` indicates that the variance is below the minimum threshold
            (i.e., uncertain), and `1.0` indicates no issue.


    Examples
    --------
    >>> time_index = pd.date_range(start='2023-01-01 00:00', periods=5, freq='30T')
    >>> s = pd.Series([0.05, 0.0, 0.1, 0.3, 0.25], index=time_index)
    >>> TSCC.detection.BASIC_byBoundaryVariance_60min(s, 0.005)
    2023-01-01 00:00:00    0.5
    2023-01-01 00:30:00    0.5
    2023-01-01 01:00:00    0.0
    2023-01-01 01:30:00    0.0
    2023-01-01 02:00:00    0.0
    Freq: 30T, Name: belowMinVar, dtype: float64
    """
    # Convert non-numeric values to NaN
    series = pd.to_numeric(series, errors='coerce')

    series.name = 'value'
    df = pd.DataFrame(series)
    df["datehour"] = df.index.strftime('%Y-%m-%d %H')
    df["belowMinVar"] = df.groupby('datehour')['value'].transform('var') < min_var

    # no error, but observations doubtful leady to change from probability 1.0 to 0.5
    return df["belowMinVar"].astype(float).replace(1.0, 0.5)


def BASIC_byStepChangeMax(series, max_diff, timestep = pd.Timedelta(minutes=30)):

    r"""
    Check if variance is too high.

    Parameters
    ----------
    series : panda series
        A series of numeric values with a timestamp index.
    max_diff : float
        The maximum allowable difference between consecutive values.
    timestep: timedelta variable, optional
        The maximum time interval between consecutive values. Default is 30 minutes.

    Returns
    -------
    pandas series
        A series where 0.5 indicates that the step change exceeds the maximum difference and 1.0 indicates no issue.

    Examples
    --------
    >>> time_index = pd.date_range(start='2023-01-01 00:00', periods=5, freq='30T')
    >>> s = pd.Series([0.05, 0.0, 3, 0.3, 0.25], index=time_index)
    >>> TSCC.detection.BASIC_byStepChangeMax(s, 2, pd.Timedelta(minutes = 30))
    2023-01-01 00:00:00    0.0
    2023-01-01 00:30:00    0.0
    2023-01-01 01:00:00    0.5
    2023-01-01 01:30:00    0.5
    2023-01-01 02:00:00    0.0
    Freq: 30T, dtype: float64
    """
    # Convert non-numeric values to NaN
    series = pd.to_numeric(series, errors='coerce')

    exceedsBoundary = (series.diff().abs() > max_diff)
    isMeaningful = (pd.Series(series.index, index = series.index).diff() <= timestep)
    isDoubtful = (exceedsBoundary & isMeaningful)
    isDoubtful.iloc[0] = False

    # no error, but observations doubtful
    return isDoubtful.astype(float).replace(1.0, 0.5)



def BASIC_byNeighbors(series, neighbors_df, max_diff=100):

    r"""
    Compare rainfall data from a given station with neighboring stations to detect outliers.

    Parameters
    ----------
    series : pandas dataframe
        A dataframe of rainfall data from a single station.
    neighbors_df : pandas.DataFrame
        A DataFrame containing rainfall data from neighboring stations.
    max_diff : float, optional
        The maximum allowable difference between the station's rainfall value and the average of neighboring stations.
        Default is 100.

    Returns
    -------
    pandas series
        A series where 0.5 indicates that the difference between the station and its neighbors exceeds
        the maximum allowable difference, and 1.0 indicates no issue.

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
    >>> TSCC.detection.BASIC_byNeighbors(df["raw"], df[["fea_1", "fea_2"]], max_diff=1)
    2023-01-01 00:00:00    0.5
    2023-01-01 00:30:00    0.5
    2023-01-01 01:00:00    0.0
    2023-01-01 01:30:00    0.0
    2023-01-01 02:00:00    0.5
    2023-01-01 02:30:00    0.5
    Freq: 30T, dtype: float64
    """

    neighbor_mean = neighbors_df.mean(axis=1)
    isDoubtful_byNeighbors = (series - neighbor_mean).abs() > max_diff

    # no error, but observations doubtful
    return isDoubtful_byNeighbors.astype(float).replace(1.0, 0.5)

def BASIC_byClimaAverage(series, period='monthly', deviation_threshold=2):

    r"""
    Perform a temporal consistency check against climatological averages (weekly, monthly, or seasonally)
    of the own time series.

    Parameters
    ----------
    series : pandas series
        A series of the target variable with a datetime index.
    period : str, optional
        The period for climatological comparison. Options are 'weekly', 'monthly', or 'seasonal'. Default is 'monthly'.
    deviation_threshold : float, optional
        The number of standard deviations from the climatological mean to flag as an error. Default is 2.

    Returns
    -------
    pandas series
        A series where 0.5 indicates that the value deviates significantly from the climatological average,
        and 1.0 indicates no significant deviation.

    Examples
    --------
    >>> time_index = pd.date_range(start='2023-01-01 00:00', periods=5, freq='30T')
    >>> s = pd.Series([0.05, 0.0, 0.1, 0.3, 0.25], index=time_index)
    >>> TSCC.detection.BASIC_byClimaAverage(s)
    index
    2023-01-01 00:00:00    0.0
    2023-01-01 00:30:00    0.0
    2023-01-01 01:00:00    0.0
    2023-01-01 01:30:00    0.0
    2023-01-01 02:00:00    0.0
    2023-01-01 02:30:00    0.0
    2023-01-01 03:00:00    0.0
    Name: clima_avg_err, dtype: float64
    """


    # Convert non-numeric values to NaN
    series = pd.to_numeric(series, errors='coerce')

    df = pd.DataFrame(series, columns = ["value"])

    # Add period columns for grouping by week, month, or season
    if period == 'weekly':
        df['period'] = df.index.isocalendar().week
    elif period == 'monthly':
        df['period'] = df.index.month
    elif period == 'seasonal':
        df['period'] = df.index.month % 12 // 3 + 1  # 1=Winter, 2=Spring, 3=Summer, 4=Fall
    else:
        raise ValueError("Invalid period type. Choose from 'weekly', 'monthly', or 'seasonal'.")

    # Calculate climatological means and standard deviations for each period
    climatology = df.groupby('period')["value"].agg(['mean', 'std']).reset_index()
    climatology.columns = ['period', 'climatology_mean', 'climatology_std']

    # Merge climatological values back to the original dataframe
    df = pd.merge(df.reset_index(), climatology, on='period', how='left').set_index("index")

    # Calculate the deviation from the climatological mean
    df['deviation'] = (df["value"] - df['climatology_mean']).abs()

    # Flag as an error if the deviation exceeds the specified number of standard deviations
    df['clima_avg_err'] = df['deviation'] > (deviation_threshold * df['climatology_std'])

    # Drop the intermediate columns (optional)
    df.drop(['period', 'climatology_mean', 'climatology_std', 'deviation'], axis=1, inplace=True)

    return df["clima_avg_err"].astype(float).replace(1.0, 0.5)

def BASIC_byPersistence(series, persistence_window=3):
    """
    Perform a persistence check to identify unrealistically constant values over consecutive time steps.

    Parameters
    ----------
    series : pandas series
        A series of the target variable with a datetime index.
    persistence_window : int, optional
        The number of consecutive time steps with the same value required to trigger an error flag. Default is 3.

    Returns
    -------
    pandas series
        A series where 0.5 indicates that the value has been constant over the specified persistence window,
        and `1.0` indicates no issue.

    Examples
    --------
    >>> time_index = pd.date_range(start='2023-01-01 00:00', periods=5, freq='30T')
    >>> s = pd.Series([0.05, 0.0, 3, 0.3, 0.25], index=time_index)
    >>> TSCC.detection.BASIC_byPersistence(s)
    2023-01-01 00:00:00    0.0
    2023-01-01 00:30:00    0.0
    2023-01-01 01:00:00    0.0
    2023-01-01 01:30:00    0.0
    2023-01-01 02:00:00    0.0
    Freq: 30T, Name: persistence_error, dtype: float64
    """

    # Convert non-numeric values to NaN
    series = pd.to_numeric(series, errors='coerce')

    df = pd.DataFrame(series, columns=["value"])

    # Sort the data by time to ensure correct chronological order
    df = df.sort_index()

    # Shift the target variable by one time step to compare consecutive values
    df['previous_value'] = df["value"].shift(1)

    # Identify consecutive time steps where the value remains the same
    df['same_value'] = (df["value"] == df['previous_value'])

    # Rolling sum to count how many consecutive steps have the same value
    df['persistence_count'] = df['same_value'].rolling(window=persistence_window, min_periods=1).sum()

    # Flag as error if the count of consecutive steps with the same value exceeds the window threshold
    df['persistence_error'] = df['persistence_count'] >= persistence_window

    # Drop intermediate columns (optional)
    df.drop(['previous_value', 'same_value', 'persistence_count'], axis=1, inplace=True)

    return df['persistence_error'].astype(float).replace(1.0, 0.5)

