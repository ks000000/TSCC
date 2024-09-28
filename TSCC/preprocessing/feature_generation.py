import pandas as pd
import numpy as np

def feature_generation_uncertainty(df, uncertainty_dict):
    r"""
    Generates uncertainty-related features and adds them to the provided dataframe.

    Parameters
    ----------
    df : pandas DataFrame
        The input dataframe to which uncertainty features will be added.
    uncertainty_dict : dict
        A dictionary where keys are feature names and values are the corresponding uncertainty of sensors
        to be added to the dataframe.

    Returns
    -------
    df : pandas DataFrame
        The dataframe with added uncertainty features based on the provided uncertainty dictionary.

    """
    for cur_key in uncertainty_dict.keys():
        df[cur_key] = uncertainty_dict[cur_key]

    return df

def feature_generation_prevObs(df, selected_columns = None, timestep_name = 5, agg_numobs = 12, agg_name = "h"):
    r"""
    Generates features based on previous observations and rolling statistics.

    Parameters
    ----------
    df : pandas DataFrame
        The input dataframe where previous observation and aggregation features will be added.
    selected_columns : list, optional
        List of specific columns for which features will be generated. If None, all columns in the dataframe are used.
    timestep_name : int, default=5
        Time interval (in terms of index steps) used for creating previous observation features.
    agg_numobs : int, default=12
        The window size for calculating rolling statistics such as min, max, mean, and sum.
    agg_name : str, default="h"
        Suffix used in the naming of the generated aggregation columns.

    Returns
    -------
    df_defragmented : pandas DataFrame
        The dataframe with additional features generated based on previous observations and rolling window statistics.

    Features Generated
    ------------------
    - Previous values (e.g., "{col}_{timestep}prev", "{col}_{2*timestep}prev")
    - Differences between the current and previous values (e.g., "{col}_{timestep}delta")
    - Rolling minimum, maximum, mean, and sum over the specified window size (e.g., "{col}_min_{agg_name}", "{col}_mean_{agg_name}")

    """

    if not selected_columns:
        initial_cols = df.columns.tolist()
    else:
        initial_cols = selected_columns

    for cur_col in initial_cols:
        df[f"{cur_col}_{timestep_name}prev"] = df[cur_col].shift(1)
        df[f"{cur_col}_{timestep_name*2}prev"] = df[cur_col].shift(2)
        df[f"{cur_col}_{timestep_name}delta"] = df[cur_col] - df[f"{cur_col}_{timestep_name}prev"]

    # not working entirely correct, probably rounding errors
    for cur_col in initial_cols:
        df[f"{cur_col}_min_{agg_name}"] = df[cur_col].rolling(agg_numobs).min()
        df[f"{cur_col}_min_6{agg_name}"] = df[cur_col].rolling(agg_numobs * 6).min()
        df[f"{cur_col}_max_{agg_name}"] = df[cur_col].rolling(agg_numobs).max()
        df[f"{cur_col}_max_6{agg_name}"] = df[cur_col].rolling(agg_numobs * 6).max()
        df[f"{cur_col}_mean_{agg_name}"] = df[cur_col].rolling(agg_numobs).mean()
        df[f"{cur_col}_mean_6{agg_name}"] = df[cur_col].rolling(agg_numobs * 6).mean()
        df[f"{cur_col}_sum_{agg_name}"] = df[cur_col].rolling(agg_numobs).sum()
        df[f"{cur_col}_sum_6{agg_name}"] = df[cur_col].rolling(agg_numobs * 6).sum()

    df_defragmented = df.copy()
    return df_defragmented

def feature_generation_time(df):
    r"""
    Generates time-based features from the index of the dataframe.

    Parameters
    ----------
    df : pandas DataFrame
        The input dataframe with a datetime index from which time-related features will be generated.

    Returns
    -------
    df : pandas DataFrame
        The dataframe with additional time-based features:
        - "month": Extracted month from the datetime index.
        - "weekday": Extracted day of the week (0 = Monday, 6 = Sunday) from the datetime index.
        - "hour": Extracted hour of the day from the datetime index.

    """

    df["month"] = df.index.month
    df["weekday"] = df.index.weekday
    df["hour"] = df.index.hour
    return df

def feature_generation_heavyRainDWD(df, selected_columns = None):
    r"""
    Generates heavy rain classification features based on rainfall thresholds defined by the DWD (German Weather Service).

    Parameters
    ----------
    df : pandas DataFrame
        The input dataframe containing rainfall data.
    selected_columns : list, optional
        List of columns to apply the heavy rain feature generation on. If None, all columns will be considered.

    Returns
    -------
    df : pandas DataFrame
        The dataframe with additional boolean columns indicating different levels of heavy rainfall:
        - "{col}_isHeavyRain": True if the rainfall in the past 1 hour is between 15-25 mm, or between 20-35 mm in the past 6 hours.
        - "{col}_isHeavyHeavyRain": True if the rainfall in the past 1 hour is between 25-40 mm, or between 35-60 mm in the past 6 hours.
        - "{col}_isExtremeHeavyRain": True if the rainfall exceeds 40 mm in the past 1 hour, or 60 mm in the past 6 hours.

    """

    for cur_col in initial_cols:
        df[f"{cur_col}_isHeavyRain"] = df[f"{cur_col}_sum_h"].between(15, 25) | df[f"{cur_col}_sum_6h"].between(20, 35)
        df[f"{cur_col}_isHeavyHeavyRain"] = df[f"{cur_col}_sum_h"].between(25, 40) | df[f"{cur_col}_sum_6h"].between(35,
                                                                                                                     60)
        df[f"{cur_col}_isExtremeHeavyRain"] = (df[f"{cur_col}_sum_h"] > 40) | (df[f"{cur_col}_sum_6h"] > 60)
        return df

def feature_generation_wrapper(df, selected_columns = None):
    r"""
    Combines various feature generation functions to augment the input dataframe with new features related to previous
    observations, seasonality, and heavy rainfall.

    Parameters
    ----------
    df : pandas DataFrame
        The input dataframe containing time series or sensor data.
    selected_columns : list, optional
        List of columns to apply the feature generation on. If None, all columns will be considered.

    Returns
    -------
    df : pandas DataFrame
        The dataframe augmented with additional features, including:
        - Previous observations and deltas using `feature_generation_prevObs`.
        - Seasonality-related features using `feature_generation_seasonality`.
        - Heavy rainfall indicators using `feature_generation_heavyRainDWD`.
    """

    df = feature_generation_prevObs(df, selected_columns)
    df = feature_generation_seasonality(df)
    df = feature_generation_heavyRainDWD(df)