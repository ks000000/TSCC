import pandas as pd
import numpy as np

#implement metrics from error classes from kiwasus additionally
#implement weighted mean as combined data quality metric

#stackoverflow, answered Oct 22, 2009 at 8:11 Ashwin Nanjappa
def isListEmpty(inList):
    if isinstance(inList, list): # Is a list
        return all( map(isListEmpty, inList) )
    return False # Not a list

def calculate_volume_indicator(n, T, delta_t, sampling_data):
    r"""
    Calculate the quality criterium data volume for a sensor network.

    Parameters
    ----------
    n : int
        Number of sensors in the network.
    T : int or float
        Monitoring time duration overall.
    delta_t : int or float
        Requested/ usual time interval of subsequent observations for each sensor.
    sampling_data : nested list
        A list of lists representing sampling data for each node.

    Returns
    -------
    qv : float
        Data volume quality indicator.

    Examples
    --------
    >>> n_nodes = 5
    >>> monitoring_duration = 5  # in arbitrary units
    >>> time_interval = 1  # in arbitrary units
    >>> node_sampling_data = [
    >>>    [1, 2, 3, None, 5],  # Example data for node 1
    >>>    [1, 2, 3, 4, 5],     # Example data for node 2
    >>>    [1, None, 3, np.NaN, 5],  # Example data for node 3
    >>>    [1, 2, 3, None, 5],     # Example data for node 4
    >>>    [1, 2, "abc", 4],     # Example data for node 5
    >>>    ]
    >>> qv_result = calculate_data_volume_indicator(n_nodes, monitoring_duration, time_interval, node_sampling_data)
    >>> qv_result
    0.8
    """

    # Function to check if observation is valid
    def f_v(data_point):
        return 1 if data_point not in [None, np.NaN] else 0

    # Calculate vi for each node
    vi_list = [sum(f_v(data_point) for data_point in node_data) for node_data in sampling_data]

    # Calculate qv
    qv = (delta_t / T) * (sum(vi_list) / n) if not isListEmpty(sampling_data) else np.NaN

    # qv is of type series type having one value in some cases
    if isinstance(qv, pd.Series):
        qv = qv[0]

    return qv

# Function to check correctness of a single sensor records
def threshold_fixed(data_sensor_x, error_threshold):
    sum = 0
    error_list = [abs(a - b) for a, b in zip(data_sensor_x[0], data_sensor_x[1])]
    for x in error_list:
        # error threshold is maximal acceptable error
        if x <= error_threshold:
            sum += 0
        else:
            sum += 1
    return sum if len(data_sensor_x) > 0 else len(data_sensor_x[0])

# Function to check correctness of a single sensor records, relative error
def threshold_rel(data_sensor_x, error_threshold):

    upperlimit_fallback = (1/error_threshold)*error_threshold
    error_threshold_fallback = error_threshold

    sum = 0
    error_list = []
    for a, b in zip(data_sensor_x[0], data_sensor_x[1]):
        # upperlimit_fallback is
        if b > upperlimit_fallback:
            cur_error = a / b
            # error threshold is maximal acceptable error
            if cur_error <= error_threshold:
                sum += 0
            else:
                sum += 1
        else:
            cur_error = abs(a - b)
            # error threshold is maximal acceptable error
            if cur_error <= error_threshold_fallback:
                sum += 0
            else:
                sum += 1

    return sum if len(data_sensor_x) > 0 else len(data_sensor_x[0])

def calculate_correctness_indicator(n, dataset, error_threshold, threshold_fct = threshold_fixed):
    r"""
    Calculate the correctness indicator for a sensor network.

    Parameters
    ----------
    n : int
        Number of nodes in the sensor network.
    dataset : list
        A list of lists representing dataset for each node.
        Every element of dataset consists of two tupel. First tupel consists of the real values
        and the second tupel consists of the observed values.
        e.g. [(25, 26, 24, 25, 26), (26, 25, 24, 25, 26)]   # [('real values'),('observed values')]
    error_threshold : int or float
        Difference between real and observed values.

    Returns
    -------
    qt : float
        Correctness Indicator.

    Examples
    --------
    >>> n_nodes = 5
    >>> error_threshold = 1.0  # Error threshold for correctness
    >>> # Example data for each node (val_real, val_observed)
    >>> node_data_sequence = [
    >>>    [(25, 26, 24, 25, 26), (26, 25, 24, 25, 26)],  # Example data for node 1
    >>>    [(20, 21, 22, 20, 21), (21, 20, 22, 20, 21)],  # Example data for node 2
    >>>    [(30, 32, 31, 30, 32), (32, 30, 31, 30, 32)],  # Example data for node 3
    >>>    [(18, 19, 18, 19, 20), (19, 18, 18, 19, 20)],  # Example data for node 4
    >>>    [(28, 29, 30, 28, 29), (29, 28, 30, 28, 29)],  # Example data for node 5
    >>>    ]
    >>> qa_result = calculate_correctness_indicator(n_nodes, node_data_sequence, error_threshold)
    >>> qa_result
    0.91999999999
    """

    # Return correct observation count for each sensor
    sum_qc_list = [threshold_fct(data_sensor_x, error_threshold) for data_sensor_x in dataset]
    # Calculate overall qa
    total_samplings = sum([len(dataset[i][0]) for i in range(len(dataset))])
    qa = 1 - (sum(sum_qc_list) / total_samplings) if not isListEmpty(dataset) else np.NaN

    #total_samplings = n * (T/delta_t)
    #total_samplings = sum([len(dataset[i][0]) for i in range(n)])#(n * T) / delta_t
    return qa

def calculate_quality_coefficients(df_list, type="all_at_once",
                                   error_threshold_corr = 0.0, threshold_fct = threshold_fixed):
    r"""
    Calculate the quality criteria for a sensor network.

    Parameters
    ----------
    df_list : list of pandas data frames
        Index for each data frame is time stamp having datetime format, first column is raw value,
        second column is ground truth value
    type : str
        Decide from ["all_at_once", "each_by_itself"].

    Returns
    -------
    q_coeff : list
        Quality indicators <volume>, <correctness> and meta data dictionary.

    Examples
    --------
    >>> df_list = []
    >>> for cur_ID in df[df_id_col].unique():
    >>>     df_list.append(df[df[df_id_col] == cur_ID].set_index("timestamp")[["value_raw", "value_plaus"]])
    >>>
    >>> qv_result = calculate_quality_coefficients(df_list)
    >>> qv_result[0]
    0.989752
    >>> qv_result[1]
    0.99
    >>> qv_result[2]
    {0: {'monitoring_duration': Timedelta('329 days 05:05:00'),
    'monitoring_duration_raw': Timedelta('329 days 05:05:00'),
    'timestep': Timedelta('0 days 00:05:00'),
    'timestep_raw': Timedelta('0 days 00:05:00')},
    1: {'monitoring_duration': Timedelta('332 days 08:15:00'),
    'monitoring_duration_raw': Timedelta('332 days 08:15:00'),
    'timestep': Timedelta('0 days 00:05:00'),
    'timestep_raw': Timedelta('0 days 00:05:00')}}
    """

    for df in df_list:
        df.columns = ["value_raw", "value_plaus"]

    monitoring_duration = pd.Series([df.index.max() - df.index.min() for df in df_list], name='monitoring_duration')
    monitoring_duration_raw = pd.Series(
        [df[["value_raw"]].dropna().index.max() - df[["value_raw"]].dropna().index.min() for df in df_list],
        name='monitoring_duration_raw')
    timestep = pd.Series([pd.Series(df.index).diff().mode()[0] for df in df_list], name='timestep')
    timestep_raw = pd.Series([pd.Series(df[["value_raw"]].dropna().index).diff().mode()[0] for df in df_list], name='timestep_raw')
    meta = pd.DataFrame(pd.concat([monitoring_duration, monitoring_duration_raw,
                                   timestep, timestep_raw], axis=1).transpose()).to_dict()
    meta.update({'n_nodes': len(df_list)})

    if type == "all_at_once":
        T = pd.Series([meta[cur_key]["monitoring_duration_raw"] for cur_key in meta.keys() if isinstance(cur_key, int)]).max()
        delta_t = pd.Series([meta[cur_key]["timestep_raw"] for cur_key in meta.keys() if isinstance(cur_key, int)]).mode()
        sampling_data = [list(df["value_raw"]) for df in df_list]
        qv = calculate_volume_indicator(meta["n_nodes"], T, delta_t, sampling_data)

        nodes_data_sequence_raw_gt = []
        for cur_key in meta.keys():
            if isinstance(cur_key, int):
                nodes_data_sequence_raw_gt.append(
                    df_list[cur_key][["value_raw", "value_plaus"]].transpose().apply(tuple, axis=1).to_list())
        #qc based on available data
        qc = calculate_correctness_indicator(meta["n_nodes"],
                                        nodes_data_sequence_raw_gt,
                                             error_threshold_corr,
                                             threshold_fct)

    elif type == "each_by_itself":
        qv = []
        for cur_key in meta.keys():
            if isinstance(cur_key, int):
                qv.append(calculate_volume_indicator(1,
                                                          meta[cur_key]["monitoring_duration_raw"],
                                                          meta[cur_key]["timestep_raw"],
                                                          [list(df_list[cur_key]["value_raw"])]))

        qc = []
        for cur_key in meta.keys():
            if isinstance(cur_key, int):
                nodes_data_sequence_raw_gt = [df_list[cur_key][["value_raw", "value_plaus"]].transpose().apply(tuple, axis=1).to_list()]
                # qc based on available data
                qc.append(calculate_correctness_indicator(meta["n_nodes"],
                                                     nodes_data_sequence_raw_gt,
                                                     error_threshold_corr,
                                                     threshold_fct))

    return qv, qc, meta


'''
def calculate_completeness_indicator(n, T, delta_t, sampling_data):
    r"""
    Calculate the Completeness Indicator (qc) for a sensor network.

    Parameters
    ----------
    n : int
        Number of nodes in the sensor network.
    T : int or float
        Monitoring time duration.
    delta_t : int or float
        Time interval for data collection.
    sampling_data : list
        A list of lists representing sampling data for each node.

    Returns
    -------
    qc : float
        Completeness Indicator.

    Examples
    --------
    >>> n_nodes = 5
    >>> monitoring_duration = 100  # in arbitrary units
    >>> time_interval = 10  # in arbitrary units
    >>> node_sampling_data = [
    >>>    [1, 2, 3, None, 5],  # Example data for node 1
    >>>    [1, 2, 3, 4, 5],     # Example data for node 2
    >>>    [1, None, 3, None, 5],  # Example data for node 3
    >>>    [1, 2, 3, 4, 5],     # Example data for node 4
    >>>    [1, 2, 3, 4, 5],     # Example data for node 5
    >>>    ]
    >>> qc_result = calculate_data_completeness_indicator(n_nodes, monitoring_duration, time_interval, node_sampling_data)
    >>> qc_result
    0.3
    """

# Soll die Funktion den ganzen Knoten dann als fehlerhaft interpretieren oder nur den einzelnen fehlerhaften Wert.
# Momentan ist es der ganze Knoten durch das break.

    # Function to check the completeness of data record X(i, t)
    def f_c(data_point):
        sum_list = 0
        for x in data_point:
            if x is None:
                sum_list = 0
                break
            else:
                sum_list += 1
        return sum_list

    # Calculate cvt for each node
    cvt_list = [f_c(data_point) for data_point in sampling_data]

    qc = (delta_t/T) * (sum(cvt_list) / n)
    # following code cannot be computed for large data sets
    # qc = (delta_t * sum(cvt_list)) / (n * T)

    return qc
    

def calculate_time_dependent_indicator(n, T, delta_t, dataset, k=1):
    r"""
    Calculate the Time Indicator (qt) for a sensor network.

    Parameters
    ----------
    n : int
        Number of nodes in the sensor network.
    T : int or float
        Monitoring time duration.
    delta_t : int or float
        Time interval for data collection.
    dataset : list
        A list of lists representing dataset for each node.
        Every list consists of the sampling data for each node and three time details.
        e.g. [(1, 2, 3, None, 5), 10, 10, 15]   #[('nodedata'),'t_ideal','t_real','t_arrive']
    k : int
        Constant for defining volatility.
        The default is 1.

    Returns
    -------
    qt : float
        Time Indicator.

    Examples
    --------
    >>> n_nodes = 5
    >>> monitoring_duration = 100  # in arbitrary units
    >>> time_interval = 10  # in arbitrary units
    >>> k_value = 1  # Constant for defining volatility

    >>> # Example data for each node (data, t_ideal, t_real, t_arrive)
    >>> node_data_sequence = [
    >>>    [(1, 2, 3, None, 5), 10, 10, 15],  # Example data for node 1
    >>>    [(1, 2, 3, 4, 5), 20, 20, 25],     # Example data for node 2
    >>>    [(1, None, 3, None, 5), 30, 30, 35],  # Example data for node 3
    >>>    [(1, 2, 3, 4, 5), 40, 40, 45],     # Example data for node 4
    >>>    [(1, 2, 3, 4, 5), 50, 50, 55],     # Example data for node 5
    >>>    ]
    >>> qt_result = calculate_time_dependent_indicator(n_nodes, monitoring_duration, time_interval, node_data_sequence, k_value)
    >>> qt_result
    0.003
    """

    # Function to calculate volatility
    def calculate_volatility(data_point):
        vola = 0
        for x in data_point:
            if x is None:
                vola = 0
                break
            else:
                vola = k * delta_t
        return vola

    # Function to calculate currency
    def calculate_currency(t_ideal, t_real, t_arrive):
        return (t_real - t_ideal) + (t_arrive - t_ideal)

    # Function to calculate time-dependent indicator for a specific node at time t
    def f_t(data_point, t_ideal, t_real, t_arrive):
        currency = calculate_currency(t_ideal, t_real, t_arrive)
        volatility = calculate_volatility(data_point)
        return max(0, 1 - currency / volatility) if volatility != 0 else 0

    # Calculate qt for each time t
    qt_list = [f_t(data_point, t_ideal, t_real, t_arrive)
                for data_point, t_ideal, t_real, t_arrive in dataset]

    # Calculate overall qt
    qt = (sum(qt_list) / (n * T)) if T > 0 else 0

    return qt
'''



