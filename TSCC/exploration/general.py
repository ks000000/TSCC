import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def aggregate_sums(df, freq="Y", date_column=None):

    r"""
    Compute sums with options for yearly, quarterly, and monthly for each column of the dataframe.

    Parameters
    ----------
    df : pandas dataframe
        A dataframe with data, index must be datetime.
    freq : str
        The frequency of aggregation. Options are ['Y' (yearly), 'Q' (quaterly), 'M'(monthly)].
    date_column : str
        Name of the column to be used as the date index. If None, the dataframe's index is used.

    Returns
    -------
    pandas dataframe
        Aggregated sums dataframe.

    Examples
    --------
    >>> # Step 1: Generate a date range for three consecutive years
    >>> date_range = pd.date_range(start='2021-01-01', end='2023-12-31', freq='D')
    >>> # Step 2: Create random data for the three columns
    >>> np.random.seed(42)  # For reproducibility
    >>> data = np.random.randint(0, 18, size=(len(date_range), 3))
    >>> # Step 3: Combine the date range and the data into a DataFrame
    >>> df = pd.DataFrame(data, index=date_range, columns=['Column1', 'Column2', 'Column3'])
    >>> TSCC.exploration.aggregate_sums(df, freq = "Y")
    	     Column1 Column2 Column3
    2021-12-31	2875	3092	2946
    2022-12-31	3189	3106	3119
    2023-12-31	2938	3115	3176
    """

    # Ensure the index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "The DataFrame index must be a DatetimeIndex or specify a date_column that can be converted to datetime.")

    # Aggregate based on the frequency
    aggregated_df = df.resample(freq).sum()

    return aggregated_df

def isCredible_ByNeighbors_yearly(series, df_neighbors, threshold = 0.1):

    r"""
    Assess credibility of sensor node throgh comparison with neighbouring sensor nodes by using the mean key figure.
    Otional function addition: distance of nodes as parameter to be respected.

    Parameters
    ----------
    series : pandas dataframe
        A dataframe of the node to be assessed. The index should be datetime-based.
    df_neighbors : pandas dataframe
        A dataframe containing time series data from neighboring sensor nodes.
        The index should match the `series` index.
    threshold : float, optional
        The acceptable deviation threshold between the node's value and the mean of its neighbors.
        The default is 0.1 (10%).

    Returns
    -------
    pandas series
        A yearly series of boolean values indicating whether the observations of the node are within the credible range
        based on its neighbors. `True` means the node is credible, and `False` indicates a significant deviation.

    Examples
    --------
    >>> # Step 1: Generate a date range for three consecutive years
    >>> date_range = pd.date_range(start='2021-01-01', end='2023-12-31', freq='D')
    >>> # Step 2: Create random data for the three columns
    >>> np.random.seed(42)  # For reproducibility
    >>> data = np.random.randint(0, 18, size=(len(date_range), 3))
    >>> # Step 3: Combine the date range and the data into a DataFrame
    >>> df = pd.DataFrame(data, index=date_range, columns=['Column1', 'Column2', 'Column3'])
    >>> TSCC.exploration.isCredible_ByNeighbors_yearly(series = df["Column1"], df_neighbors = df[["Column2", "Column3"]])
    2021    True
    2022    True
    2023    True
    dtype: bool
    """

    series.name = "to_be_checked"
    df_series = pd.DataFrame(series)
    df_series = df_series.join(df_neighbors)
    df_series = df_series.groupby(df_series.index.year).mean()
    return (abs(df_series["to_be_checked"] - df_series.drop(columns = ["to_be_checked"]).mean(axis = 1))/df_series["to_be_checked"]) <= threshold


def getGapDist(series):

    r"""
    Calculate and summarize the distribution of gaps between consecutive time index values in a time series.

    Parameters
    ----------
    series : pandas series
        The time series with a datetime index for which the gap distribution is to be calculated.

    Returns
    -------
    pandas series
        A series representing the distribution of time gaps between consecutive index values,
        where the index is the gap (timedelta) and the values are the counts of how often each gap occurs.

    Examples
    --------
    >>> # Step 1: Generate a date range for three consecutive years
    >>> date_range = pd.date_range(start='2021-01-01', end='2023-12-31', freq='D')
    >>> # Step 2: Create random data for the three columns
    >>> np.random.seed(42)  # For reproducibility
    >>> data = np.random.randint(0, 18, size=(len(date_range), 3))
    >>> # Step 3: Combine the date range and the data into a DataFrame
    >>> df = pd.DataFrame(data, index=date_range, columns=['Column1', 'Column2', 'Column3'])
    >>> TSCC.exploration.getGapDist(df["Column1"])
    1 days    1094
    Name: count, dtype: int64
    """

    # Calculate the differences between consecutive time index values
    time_diffs = series.index.to_series().diff().dropna()

    # Summarize the differences
    gap_distribution = time_diffs.value_counts().sort_index()

    return gap_distribution


def getDoubleMassAnalysis(df, target_var, reference_vars, plot=True):

    r"""
    Perform Double Mass Analysis to assess the consistency of a target variable (e.g., rainfall).

    Parameters
    ----------
    df : pandas dataframe
        A dataframe containing the target variable, reference variables, and a datetime index or column.
    target_var : str
        The target variable (e.g., 'rainfall') whose consistency is being checked.
    reference_vars : list of str
        A list of reference variables, typically from nearby stations, to compare with the target variable.
    plot : bool, optional
        If True, a plot of the Double Mass Curve will be generated. The default is True.

    Returns
    -------
    pandas dataframe
        A dataframe containing the cumulative sums of the target variable and the reference variables.

    Examples
    --------
    >>> # Step 1: Generate a date range for three consecutive years
    >>> date_range = pd.date_range(start='2021-01-01', end='2023-12-31', freq='D')
    >>> # Step 2: Create random data for the three columns
    >>> np.random.seed(42)  # For reproducibility
    >>> data = np.random.randint(0, 18, size=(len(date_range), 3))
    >>> # Step 3: Combine the date range and the data into a DataFrame
    >>> df = pd.DataFrame(data, index=date_range, columns=['Column1', 'Column2', 'Column3'])
    >>> TSCC.exploration.getDoubleMassAnalysis(df, "Column1", ["Column2", "Column3"])
        cum_target	cum_reference
    2021-01-01	6	12.0
    2021-01-02	13	20.0
    2021-01-03	23	25.0
    2021-01-04	25	31.0
    2021-01-05	30	31.5
    ...	...	...
    2023-12-27	8958	9254.0
    2023-12-28	8974	9255.5
    2023-12-29	8990	9260.0
    2023-12-30	8997	9270.0
    2023-12-31	9002	9277.0
    1095 rows × 2 columns
    """


    # Sort the data by time to ensure correct chronological order
    df = df.sort_index()

    # Calculate the cumulative sum of the target variable
    df['cum_target'] = df[target_var].cumsum()

    # Calculate the cumulative sum of the reference variables (mean of reference series)
    df['cum_reference'] = df[reference_vars].mean(axis=1).cumsum()

    # Plot the Double Mass Curve if requested
    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(df['cum_reference'], df['cum_target'], label=f'{target_var} vs. Reference', marker='o')
        plt.xlabel('Cumulative Reference (Mean of Reference Variables)')
        plt.ylabel(f'Cumulative {target_var}')
        plt.title('Double Mass Analysis')
        plt.grid(True)
        plt.legend()
        plt.show()

    return df[['cum_target', 'cum_reference']]

def getCorrelationAnalysis(df, target_col, other_cols=[], plot=True):

    r"""
    Perform correlation analysis between the target variable and other meteorological variables.

    Parameters
    ----------
    df : dataframe
        The dataframe contains the target variable and other meteorological variables.
    target_col : str
        The target variable (e.g., 'rainfall') to correlate with other variables.
    other_cols : list of str
        A list of other meteorological variables to check correlation against.
    plot : bool, optional
        If True, plot the correlation matrix as a heatmap. The default is True.

    Returns
    -------
    pandas dataframe
        A dataframe containing correlation values between target and other variables.

    Examples
    --------
    >>> # Step 1: Generate a date range for three consecutive years
    >>> date_range = pd.date_range(start='2021-01-01', end='2023-12-31', freq='D')
    >>> # Step 2: Create random data for the three columns
    >>> np.random.seed(42)  # For reproducibility
    >>> data = np.random.randint(0, 18, size=(len(date_range), 3))
    >>> # Step 3: Combine the date range and the data into a DataFrame
    >>> df = pd.DataFrame(data, index=date_range, columns=['Column1', 'Column2', 'Column3'])
    >>> TSCC.exploration.getCorrelationAnalysis(df, "Column1", ["Column2", "Column3"])
             Column1	Column2	     Column3
    Column1	1.000000	0.01820	    0.038464
    Column2	0.018200	1.00000	    0.021740
    Column3	0.038464	0.02174	    1.000000
    """

    # Select only the relevant columns for correlation
    selected_vars = [target_col] + other_cols
    correlation_matrix = df[selected_vars].corr()

    if plot:
        # Plot correlation matrix as a heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(f'Correlation Matrix of {target_col} and Other Variables')
        plt.show()

    return correlation_matrix

def identify_stuckAtZero(df):
    drop_cols_afterwards = []
    for i in range(6):
        df[f"val_raw_{(i + 1) * 5}prev"] = df["val_raw"].shift(i + 1)
        drop_cols_afterwards.append(f"val_raw_{(i + 1) * 5}prev")

    df["isStuckAtZero"] = ((df["error_class"].isna()) &
                           (df["val_raw"] == 0) &
                           (df["val_gt"] > 0) &
                           (df["val_raw_5prev"] == 0) &
                           (df["val_raw_10prev"] == 0) &
                           (df["val_raw_15prev"] == 0) &
                           (df["val_raw_20prev"] == 0) &
                           (df["val_raw_25prev"] == 0) &
                           (df["val_raw_30prev"] == 0))

    df.loc[df["isStuckAtZero"] == True, "error_class"] = 'StuckAtZero'

    df = df.drop(columns=["isStuckAtZero"] + drop_cols_afterwards)

    return df


def identify_outlier(df, uncertainty_threshold = 0.1, outlier_distance = 10):

    df["val_raw_kernelmean_l"] = df["val_raw"].rolling(window=12, center=False, closed='left').mean()
    df["val_raw_kernelstd_l"] = df["val_raw"].rolling(window=12, center=False, closed='left').std()

    df["val_raw_kernelmean_r"] = df["val_raw_kernelmean_l"].shift(-13)
    df["val_raw_kernelstd_r"] = df["val_raw_kernelstd_l"].shift(-13)
    df["val_raw_kernelmean"] = df[["val_raw_kernelmean_l", "val_raw_kernelmean_r"]].mean(axis=1, skipna=False)
    df["val_raw_kernelstd"] = df[["val_raw_kernelstd_l", "val_raw_kernelstd_r"]].mean(axis=1, skipna=False)

    df["isOutlier"] = (df["error_class"].isna() &
                       ((abs(df["val_raw"] - df["val_raw_kernelmean"]) > df[
                           "val_raw_kernelstd"] * outlier_distance) &
                        (abs(df["val_raw"] - df["val_raw_kernelmean"]) > uncertainty_threshold)))

    df.loc[df["isOutlier"] == True, "error_class"] = 'Outlier'

    df = df.drop(columns=
                 ["isOutlier",
                  "val_raw_kernelmean_l", "val_raw_kernelstd_l",
                  "val_raw_kernelmean_r", "val_raw_kernelstd_r",
                  "val_raw_kernelmean", "val_raw_kernelstd"])
    return df


def identify_drift(df):
    drop_cols_afterwards = []
    for i in range(2):
        df[f"val_raw_{(i + 1) * 5}prev"] = df["val_raw"].shift(i + 1)
        drop_cols_afterwards.append(f"val_raw_{(i + 1) * 5}prev")
        df[f"error_class_{(i + 1) * 5}prev"] = df["error_class"].shift(i + 1)
        drop_cols_afterwards.append(f"error_class_{(i + 1) * 5}prev")

    df["error_class_cur_prev"] = (
                df["error_class"].isna() & df["error_class_5prev"].isna() & df["error_class_10prev"].isna())
    df["isUpperDrift"] = (df["error_class_cur_prev"].isna() &
                          ((df["val_raw"] < df["val_raw_5prev"]) &
                           (df["val_raw_5prev"] < df["val_raw_10prev"])))
    df["isLowerDrift"] = (df["error_class_cur_prev"].isna() &
                          ((df["val_raw"] > df["val_raw_5prev"]) &
                           (df["val_raw_5prev"] > df["val_raw_10prev"])))
    df["isDrift"] = df["isUpperDrift"] | df["isLowerDrift"]

    df.loc[df["isDrift"] == True, "error_class"] = 'Drift'

    df = df.drop(columns=["isDrift", "isUpperDrift", "isLowerDrift", "error_class_cur_prev"] + drop_cols_afterwards)

    return df

'''
def identify_constantValue(df):
    drop_cols_afterwards = []
    df[f"val_raw_{(0) * 5}prev"] = df["val_raw"]
    drop_cols_afterwards.append(f"val_raw_{(0) * 5}prev")
    for i in range(6):
        df[f"val_raw_{(i + 1) * 5}prev"] = df["val_raw"].shift(i + 1)
        df[f"delta_{(i) * 5}_to_{(i + 1) * 5}"] = abs(
            df[f"val_raw_{(i + 1) * 5}prev"] - df[f"val_raw_{(i) * 5}prev"])
        drop_cols_afterwards.append(f"val_raw_{(i + 1) * 5}prev")
        drop_cols_afterwards.append(f"delta_{(i) * 5}_to_{(i + 1) * 5}")

    delta_cols = ["delta_0_to_5", "delta_5_to_10",
                  "delta_10_to_15"]  # , "delta_15_to_20", "delta_20_to_25", "delta_25_to_30"]
    df["isConstant"] = ((df["error_class"].isna()) &
                        (df[delta_cols] == 0).sum(axis=1, skipna=False) == len(delta_cols))

    df.loc[df["isConstant"] == True, "error_class"] = 'Constant'

    df = df.drop(columns=["isConstant"] + drop_cols_afterwards)

    return df
'''


def identify_constantValue(df, window):
    df['isConstant_alsoNoError'] = df["val_raw"].rolling(window=window, min_periods=1).apply(lambda x: len(set(x)) == 1,
                                                                                            raw=True)
    df['isConstant'] = df['isConstant_alsoNoError'].astype(bool) & (df["val_raw"] != df["val_gt"])

    return df

def identify_uncertainty(df, uncertainty_threshold = 0.1):
    # sensor uncertainty
    df["isUncertain"] = (df["error_class"].isna() &
                         (abs(df["val_raw"] - df["val_gt"]) <= uncertainty_threshold))

    df.loc[df["isUncertain"] == True, "error_class"] = 'Uncertainty'
    df = df.drop(columns=["isUncertain"])

    return df


def identify_errorClasses(series_raw, series_gt, uncertainty_threshold = 0.1):
    r"""
    This function compares raw values (`series_raw`) with ground truth values (`series_gt`) and assigns an error class for each
    observation. It identifies specific error types, including 'Not evaluable' for missing values in the ground truth,
    'No error' for matching values, and 'Missing value' where the raw value is missing but the ground truth is present.
    Additionally, the function integrates several error detection functions (e.g., `identify_stuckAtZero`, `identify_outlier`,
    `identify_drift`, `identify_constantValue`, `identify_uncertainty`) to flag other error patterns, such as constant values
    or outliers. The final output is a series of categorized error classes.

    Parameters
    ----------
    series_raw : pandas Series
        The raw data series.
    series_gt : pandas Series
        The ground truth data series.

    Returns
    -------
    error_class : pandas Series
        A categorical series indicating the type of error for each observation.

    """

    df = pd.DataFrame([series_raw, series_gt]).transpose()

    # Define error classes
    df["error_class"] = np.nan
    df.columns = ["val_raw", "val_gt", "error_class"]

    df["isNotEvaluated"] = df["val_gt"].isna() #& df["val_raw"].isna()
    df.loc[df["isNotEvaluated"] == True, "error_class"] = 'Not evaluable'
    df = df.drop(columns=["isNotEvaluated"])

    df["isNoError"] = (df["val_gt"] == df["val_raw"])
    df.loc[df["isNoError"] == True, "error_class"] = 'No error'
    df = df.drop(columns=["isNoError"])

    df["isMissingValue"] = (~df["val_gt"].isna())& df["val_raw"].isna()
    df.loc[df["isMissingValue"] == True, "error_class"] = 'Missing value'
    df = df.drop(columns=["isMissingValue"])

    # order is important, error classes are not disjoint
    df = identify_stuckAtZero(df)
    df = identify_constantValue(df, 4)
    df = identify_outlier(df)
    df = identify_drift(df)
    df = identify_uncertainty(df, uncertainty_threshold = uncertainty_threshold)

    df.loc[df["error_class"].isna(), "error_class"] = 'Unidentified error'

    df["error_class"] = df["error_class"].astype("category")

    return df["error_class"]




