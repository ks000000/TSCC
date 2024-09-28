import pandas as pd
import numpy as np
import random


def get_indices(cur_row, timedelta):
    # helper function
    if cur_row.poissonValue == 1:
        return [cur_row.timestamp]
    else:
        timestamp_consecutive = [cur_row.timestamp ] *cur_row.poissonValue
        for i, cur_ts in enumerate(timestamp_consecutive):
            timestamp_consecutive[i] = cur_ts + i *timedelta
        timestamp_consecutive
        return timestamp_consecutive


def remove_percentage_of_timestamps(timestamps, percentage):
    # Determine the number of observations to remove
    num_to_remove = int(len(timestamps) * percentage / 100)

    # Randomly select indices to remove
    indices_to_remove = np.random.choice(len(timestamps), num_to_remove, replace=False)

    # Remove the selected observations from the array
    remaining_timestamps = [timestamps[i] for i in range(len(timestamps)) if i not in indices_to_remove]

    return remaining_timestamps

def get_index_pois(list_index, timedelta, lambda_pois = 1):
    # helper function
    list_index.sort()
    list_index = remove_percentage_of_timestamps(list_index, (1-1/lambda_pois)*100)
    # number of starting timestamps is number targetet elements / expected value of obs per starting timstamp
    #help_indices = pd.DataFrame([list_index, np.random.poisson(lambda_pois, int(len(list_index)/lambda_pois))]).transpose()
    help_indices = pd.DataFrame([list_index, np.random.poisson(lambda_pois, len(list_index))]).transpose()
    help_indices.columns = ["timestamp", "poissonValue"]
    help_indices = help_indices[help_indices["poissonValue" ] >0]
    help_indices["timestamp_consecutive"] = help_indices.apply(
        lambda cur_row: get_indices(cur_row, timedelta), axis = 1)
    list_index_pois = list(help_indices["timestamp_consecutive"].explode())

    help_indices = help_indices.reset_index(drop = True).reset_index()
    help_indices["group"] = help_indices.apply(lambda cur_row: [cur_row["index"] ] *cur_row.poissonValue, axis = 1)
    list_index_group = list(help_indices["group"].explode())

    return list_index_pois, list_index_group  # , help_indices

def get_rand_bias(n, int_max):
    # helper function
    rand_list = []
    for i in range(n):
        # random.randrange(0, 2) -> returns one from {0,1}
        rand_binary = random.randrange(0, 2)
        # random number in interval [0,1] multiplied with maximal value of series or
        # a random number within whole range of series is reached
        range_rand = random.randrange(-int(int_max*1.1), int(int_max*1.1)) if int_max != 0 else 0
        rand_list.append(rand_binary *random.random( ) *int_max + \
                         ((rand_binary + 1 ) %2 ) * range_rand)
    return rand_list


def generateSyntheticErrors(s, error_type, error_rate, timedelta = None,
                            random_seed = 1, lambda_arr = [4, 5]):
    r"""
    Generate synthetic errors in a time series data set based on specified error types.

    Parameters
    ----------
    s : pandas Series
        Series with a timestamp index where synthetic errors are to be generated.
    error_type : list
        List of error types to apply. Options include:
        "noise", "bias", "drift", "constant value", "outlier", and "missing".
    error_rate : float
        Overall error rate as a value between (0, 1], distributed equally across the error types.
    timedelta : timedelta, optional
        Time step of the series. If not provided, it is calculated from the index.
    random_seed : int, optional
        Seed for random number generation to ensure reproducibility, default is 1.
    lambda_arr : array, optional
        Array specifying lambdas for certain error types, where:
        - First value is used for bias errors
        - Second value is used for constant value errors
        default is [4, 5].

    Returns
    -------
    s : pandas Series
        The series with synthetic errors injected based on the specified error types.
    s_etype : pandas Series
        A series of the same length as `s`, indicating the error type for each observation.

    Examples
    --------
    >>> import numpy as np

    >>> num_samples = 10
    >>> s = pd.Series(np.random.normal(0, 5, num_samples), name='initial')
    >>> s_err, s_errtype = TSCC.preprocessing.generateSyntheticErrors(s, error_type = ["noise"], error_rate = 0.5)
    >>> s_err.name = "with_errors"
    >>> print(pd.DataFrame([s, s_err]).transpose())
         initial  with_errors
    0   6.249285    -1.482565
    1  -9.831159    -5.744147
    2 -11.766656   -11.766656
    3  -5.868419   -10.010263
    4  -1.630816    -1.630816
    5  -2.720739    -2.720739
    6   3.339932     3.339932
    7  -6.249174    -6.249174
    8  -4.315455    -4.315455
    9  -8.631155   -15.482079
    """

    s = s.copy(deep = True)

    # random_state = 5
    random.seed(random_seed)

    # error rate of each error type
    error_rate_each = error_rate /len(error_type)
    if error_rate_each*len(s) < 1.0:
        print("Series is too short.")
        return None, None

    lambda_bias, lambda_const = lambda_arr

    if not timedelta:
        timedelta = s.index[1] - s.index[0]

    # series describing generated error types for observations
    s_etype = pd.Series("", index = s.index, name = "error_type")

    if "noise" in error_type:

        noise_function = np.random.normal

        list_index_single = list(s.sample(frac = error_rate_each).index)
        list_index, l_gr = get_index_pois(list_index_single, timedelta)

        # only keep elements that exist in original series
        list_index = [i for i in list_index if i in list(s.index)]

        # use standard deviation of series for gaussian noise
        # s = s.where(~s.index.isin(list_index), s + noise_function(0, s.std(), 1))
        s[s.index.isin(list_index)] = s[s.index.isin(list_index)] + noise_function(0, s.std(), len(s[s.index.isin(list_index)]))
        s_etype = s_etype.where(~s_etype.index.isin(list_index), "noise")

    if "bias" in error_type:

        list_index_single = list(s.sample(frac = error_rate_each).index)
        list_index, l_gr = get_index_pois(list_index_single, timedelta, lambda_pois = lambda_bias)

        # only keep elements that exist in original series
        list_index = [i for i in list_index if i in list(s.index)]

        df_tmp = pd.DataFrame([list_index, l_gr]).transpose()
        df_tmp.columns = ["timestamp", "groupNr"]
        df_tmp = df_tmp.groupby(["groupNr"])["timestamp"].apply(list)
        df_tmp = pd.DataFrame(df_tmp)
        df_tmp["bias"] = get_rand_bias(df_tmp.shape[0], int(s.quantile(.5)))
        df_tmp["tmp_count"] = df_tmp["timestamp"].apply(lambda x: len(x))
        df_tmp["bias_n"] = df_tmp.apply(lambda cur_row: [cur_row.bias ] *cur_row.tmp_count, axis = 1)
        df_tmp2 = pd.DataFrame([list_index, list(df_tmp["bias_n"].explode())]).transpose().drop_duplicates \
            (subset = [0]).dropna()
        list_biases = list(df_tmp2[1])
        list_index = list(df_tmp2[0])
        s[s.index.isin(list_index)] = s[s.index.isin(list_index)] + list_biases
        s_etype = s_etype.where(~s_etype.index.isin(list_index), "bias")

    if "drift" in error_type:
        # not implemnted yet
        pass

    if "constant value" in error_type:

        list_index_single = list(s.sample(frac = error_rate_each).index)
        list_index, l_gr = get_index_pois(list_index_single, timedelta, lambda_pois = lambda_const)

        # only keep elements that exist in original series
        list_index = [i for i in list_index if i in list(s.index)]

        df_tmp = pd.DataFrame([list_index, l_gr]).transpose()
        df_tmp.columns = ["timestamp", "groupNr"]
        df_tmp = df_tmp.groupby(["groupNr"])["timestamp"].apply(list)
        df_tmp = pd.DataFrame(df_tmp)
        df_tmp["bias"] = get_rand_bias(df_tmp.shape[0], int(s.quantile(.5)))
        df_tmp["tmp_count"] = df_tmp["timestamp"].apply(lambda x: len(x))
        df_tmp["bias_n"] = df_tmp.apply(lambda cur_row: [cur_row.bias ] *cur_row.tmp_count, axis = 1)
        df_tmp2 = pd.DataFrame([list_index, list(df_tmp["bias_n"].explode())]).transpose().drop_duplicates \
            (subset = [0]).dropna()
        list_biases = list(df_tmp2[1])
        list_index = list(df_tmp2[0])
        s[s.index.isin(list_index)] = list_biases
        s_etype = s_etype.where(~s_etype.index.isin(list_index), "constant value")

    if "outlier" in error_type:

        # outliers as single events, no consecutive outliers are generated
        list_index = list(s.sample(frac = error_rate_each).index)

        # only keep elements that exist in original series
        list_index = [i for i in list_index if i in list(s.index)]

        s = s.where(~s.index.isin(list_index), s + random.random( ) *s.quantile(.95))
        s_etype = s_etype.where(~s_etype.index.isin(list_index), "outlier")

    if "missing" in error_type:

        list_index_single = list(s.sample(frac = error_rate_each).index)
        list_index, l_gr = get_index_pois(list_index_single, timedelta)

        # only keep elements that exist in original series
        list_index = [i for i in list_index if i in list(s.index)]

        # if condition of where statement is false, replace value with nan
        s = s.where(~s.index.isin(list_index), np.NaN)
        s_etype = s_etype.where(~s_etype.index.isin(list_index), "missing")

    return s, s_etype