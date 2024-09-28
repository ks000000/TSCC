import pandas as pd
import numpy as np

def fct_indicator(x, threshold_min, threshold_max):
    '''
    for internal use only
    '''
    return pd.Series(np.where(x.between(threshold_min, threshold_max, inclusive='both'), 1, 0), index=x.index)

def getEvents(s, max_time_dist_from_center,
               center_valley = (-float("inf"), float("inf")),
               center_peak = (-float("inf"), float("inf")),
              event_sum = (-float("inf"), float("inf")),
              event_dist = (-float("inf"), float("inf"), 0.0)):
    r"""
    Finds events that satisfy the following conditions: A minimal intensity,
    a minimal and maximal time span of the event of event.

    Parameters
    ----------
    s : pandas series
        Series with numerical entries, index in datetime format
        timestep must be equidistant
    max_time_dist_from_center : string
        Defines the timestep lengths of the maximal distance to the center of the event
        Has to be the format number + letter.
        Possible for time step units (letter):
            seconds, minutes, hours, days, years.
        Examples: "2s" for two seconds, "5t" for five minute,
                  "5h" for five hours, "4d" for four day or "1y" for one year.
    center_valley : optional, numeric duple
        The min and max valley intensity that an event must have
    center_peak : optional, numeric duple
        The min and max peak intensity that an event must have
    event_sum : optional, numeric duple
        The min and max event sum that an event must have
    event_dist : (min, max, rate_included)
        The event range by min and max value combined with the
        rate range in [0,1], stating the value rate outside the typical event interval

    Returns
    -------
    list
        nested list with a sublist of each event including the
        - starting time
        - ending time
        - minimal intensity
        - maximal intensity
        - event sum
        in that order.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Create a date range for the timestamps
    >>> date_range = pd.date_range(start='2000-04-29', periods=50, freq='5min')
    >>>
    >>> # Generate random data for 'value_plaus'
    >>> np.random.seed(0)  # For reproducibility
    >>> value_plaus = np.random.poisson(lam=1, size=len(date_range))  # Lambda set to 3
    >>>
    >>> # Create the DataFrame
    >>> df = pd.DataFrame({
    >>>     'timestamp': date_range,
    >>>     'value_plaus': value_plaus
    >>> })
    >>>
    >>> event_list = TSCC.exploration.getEvents(df.set_index("timestamp")["value_plaus"],
    >>>     max_time_dist_from_center = "45t",
    >>>     #center_valley = (-float("inf"), float("inf")),
    >>>     center_peak = (2, float("inf")),
    >>>     event_sum=(3, 140),
    >>>     event_dist = (0.000001, float("inf"), .3))
    >>> print(event_list)
    [[Timestamp('2000-04-29 00:00:00'),
      Timestamp('2000-04-29 00:45:00'),
      0,
      5,
      15],
     [Timestamp('2000-04-29 00:10:00'),
      Timestamp('2000-04-29 01:40:00'),
      0,
      5,
      24],
     [Timestamp('2000-04-29 02:15:00'),
      Timestamp('2000-04-29 03:45:00'),
      0,
      3,
      13],
     [Timestamp('2000-04-29 03:05:00'),
      Timestamp('2000-04-29 04:05:00'), 0, 2, 9]]

    """

    df_fct = pd.DataFrame(s.copy(deep=True)).reset_index()
    df_fct.columns = ["timestamp", "value"]
    df_fct = df_fct.sort_values(by="timestamp")

    # check for intensities (for each observation)
    df_fct["kept_peak_limits"] = df_fct["value"].between(center_peak[0], center_peak[1])
    df_fct["kept_valley_limits"] = df_fct["value"].between(center_valley[0], center_valley[1])
    # extract center times of possible events
    possible_event_center_time = df_fct[df_fct["kept_peak_limits"] & df_fct["kept_valley_limits"]]["timestamp"]

    event_list = []
    for cur_timestamp in possible_event_center_time:
        if event_list:
            # dont overlap events, if at least one event is in list
            if cur_timestamp <= event_list[-1][1]:
                continue
        # extract possible timeframe to left and right
        df_event = df_fct.query(
            f'"{cur_timestamp - pd.Timedelta(max_time_dist_from_center)}" <= timestamp <= "{cur_timestamp + pd.Timedelta(max_time_dist_from_center)}"')
        df_event = df_event.set_index("timestamp")
        # exclude beginning and end of event with 0 values
        #df_event = df_event[df_event["value"].cumsum() > 0]
        #df_event = df_event[df_event.loc[::-1, "value"].cumsum()[::-1] > 0]
        # compute key figures for event
        ev_peak_intensity = df_event["value"].max()
        ev_valley_intensity = df_event["value"].min()
        max_intensity_idx = df_event["value"].idxmax()
        ev_event_sum = df_event["value"].sum()
        # check conditions for event
        if not (event_sum[0] <= ev_event_sum <= event_sum[1]):
            # skip this possible event due to wrong event sum
            continue
        if not ((fct_indicator(df_event["value"], event_dist[0], event_dist[1]).sum() / df_event.shape[0]) >= event_dist[2]):
            # skip this possible event due to wrong distribution of data
            continue
        event_list.append([
            min(max_intensity_idx, df_event.index.min()),
            max(max_intensity_idx, df_event.index.max()),
            ev_valley_intensity,
            ev_peak_intensity,
            ev_event_sum])

    return event_list


def setDictRect(x0, x1):
    '''
    for internal use only

    The function generates a dictionary with characteristics of a rectangle.
    '''

    return dict(fillcolor="rgba(200, 0, 0, 0.4)",
                line={"width": 0},
                type="rect",
                x0=x0,      x1=x1,
                xref="x",   yref="paper",
                y0=0.01,    y1=0.99,
                )


def indices2dict(K, J):
    '''
    for internal use only
    '''

    return [setDictRect(J[t0], J[t1]) for t0, t1 in K]


def addShapes(K, J, fig):
    '''
    for internal use only
    '''

    fig.update_layout(
        shapes=indices2dict(K, J)
    )
    return fig


def highlightEventsPlot(s, event_list):
    r"""
    Highlights specific events on a time series plot using Plotly.

    Parameters
    ----------
    s : pandas Series
        A series with a datetime index representing a time series.
    event_list : nested list
        A list containing event intervals, where each event is defined by two
        timestamps (start and end). This is typically the output from the
        getEvents() function.

    Returns
    -------
    plot : Plotly plot
        A time series plot with highlighted event intervals.

    Examples
    --------
    # Assuming the DataFrame 'df' has a 'timestamp' column and a 'value_plaus' column:

    >>> df_time = "timestamp"
    >>> df_value_raw = "value_plaus"
    >>> # Get the list of events using the getEvents() function:
    >>> event_list = getEvents(df.iloc[:60000].set_index(df_time)[df_value_raw],
    >>>                        max_time_dist_from_center="45t",
    >>>                        center_peak=(2, float("inf")),
    >>>                        event_sum=(3, 140),
    >>>                        event_dist=(0.000001, float("inf"), .3))
    >>> # Plot the time series with highlighted events:
    >>> highlightEventsPlot(df.set_index("timestamp").iloc[:60000]["value_plaus"],
    >>>                     event_list)
    """

    import plotly.graph_objects as go
    import plotly.io as pio

    df = pd.DataFrame(s).reset_index(names = "timestamp")

    K = [[df.timestamp[df.timestamp == cur_timestamp[0]].index.values[0],
          df.timestamp[df.timestamp == cur_timestamp[1]].index.values[0]] for cur_timestamp in event_list]

    df = df.set_index("timestamp")

    pio.renderers.default = 'browser'
    fig = go.Figure()
    fig = fig.add_trace(go.Scatter(x=df.index,
                                   y=df.values.reshape(1, -1)[0],
                                   ))
    fig.update_layout(
        shapes=indices2dict(K, df.index)
    )
    fig.show()


def getEventReturnPeriod(series, KOSTRA_DS):
    r"""
    Calculate heavy rainfall events based on rolling sums over different durations
    and thresholds from the KOSTRA_DS dataset.

    Parameters
    ----------
    series : pandas series
        A time series of rainfall data with a datetime index and rainfall amounts as values.
    KOSTRA_DS : pandas dataframe
        A DataFrame containing threshold values from the KOSTRA-DS dataset.
        Columns represent different probabilities (`wdkzeit`), and rows represent
        durations (`duration_level`) for heavy rainfall.

    Returns
    -------
    pandas dataframe
        A DataFrame where each row corresponds to a timestamp and includes binary
        indicators for whether heavy rainfall occurred for each probability (`wdkzeit`).

    Example
    -------
    >>>import pandas as pd
    >>>
    >>># Define the data
    >>>data = {
    >>>    "1a": [9.7, 12.0, 14.5, 16.1, 17.4, 20.8, 23.0, 27.4, 30.4, 32.6, 38.8, 43.0],
    >>>    "3a": [13.0, 16.0, 19.4, 21.6, 23.2, 27.8, 30.8, 36.7, 40.6, 43.6, 51.9, 57.5],
    >>>    "5a": [14.7, 18.0, 21.8, 24.3, 26.2, 31.3, 34.7, 41.3, 45.7, 49.2, 58.5, 64.7],
    >>>    "10a": [17.0, 20.9, 25.3, 28.2, 30.4, 36.3, 40.3, 48.0, 53.1, 57.1, 67.9, 75.2],
    >>>    "20a": [19.5, 24.0, 29.0, 32.3, 34.8, 41.5, 46.1, 54.9, 60.7, 65.3, 77.7, 86.0],
    >>>    "25a": [20.4, 25.0, 30.3, 33.7, 36.3, 43.4, 48.1, 57.3, 63.5, 68.2, 81.2, 89.9],
    >>>    "30a": [21.1, 25.9, 31.4, 34.9, 37.6, 45.0, 49.8, 59.4, 65.7, 70.7, 84.1, 93.1],
    >>>    "50a": [23.2, 28.5, 34.5, 38.4, 41.4, 49.4, 54.8, 65.3, 72.3, 77.7, 92.5, 102.4],
    >>>    "100a": [26.2, 32.2, 39.0, 43.4, 46.7, 55.9, 61.9, 73.7, 81.7, 87.8, 104.5, 115.6]
    >>>}
    >>>
    >>># Define the index (time intervals)
    >>>index = ["15 min", "30 min", "60 min", "90 min", "2 h", "4 h", "6 h", "12 h", "18 h", "24 h", "48 h", "72 h"]
    >>>
    >>># Create the DataFrame
    >>>df = pd.DataFrame(data, index=index)
    >>>
    >>># Display the DataFrame
    >>>print(df)
    >>>
    >>> # Get a DataFrame with heavy rainfall event indicators
    >>> df_re = TSCC.exploration.getEventReturnPeriod(df.set_index(df_time)[df_value_raw], KOSTRA_MH)
    >>>print(df_re.head())
                             isHeavyRain_AtLeast1a  isHeavyRain_AtLeast3a  \
    timestamp
    2000-04-29 00:00:00                  False                  False
    2000-04-29 00:05:00                  False                  False
    2000-04-29 00:10:00                  False                  False
    2000-04-29 00:15:00                  False                  False
    2000-04-29 00:20:00                  False                  False

                         isHeavyRain_AtLeast5a  isHeavyRain_AtLeast10a  \
    timestamp
    2000-04-29 00:00:00                  False                   False
    2000-04-29 00:05:00                  False                   False
    2000-04-29 00:10:00                  False                   False
    2000-04-29 00:15:00                  False                   False
    2000-04-29 00:20:00                  False                   False

                         isHeavyRain_AtLeast20a  isHeavyRain_AtLeast25a  \
    timestamp
    2000-04-29 00:00:00                   False                   False
    2000-04-29 00:05:00                   False                   False
    2000-04-29 00:10:00                   False                   False
    2000-04-29 00:15:00                   False                   False
    2000-04-29 00:20:00                   False                   False

                         isHeavyRain_AtLeast30a  isHeavyRain_AtLeast50a  \
    timestamp
    2000-04-29 00:00:00                   False                   False
    2000-04-29 00:05:00                   False                   False
    2000-04-29 00:10:00                   False                   False
    2000-04-29 00:15:00                   False                   False
    2000-04-29 00:20:00                   False                   False

                         isHeavyRain_AtLeast100a
    timestamp
    2000-04-29 00:00:00                    False
    2000-04-29 00:05:00                    False
    2000-04-29 00:10:00                    False
    2000-04-29 00:15:00                    False
    2000-04-29 00:20:00                    False
    """

    df = pd.DataFrame(series)
    df[f"amount_15 min"] = series.rolling(3).sum()
    df[f"amount_30 min"] = series.rolling(6).sum()
    df[f"amount_60 min"] = series.rolling(12).sum()
    df[f"amount_90 min"] = series.rolling(18).sum()
    df[f"amount_2 h"] = series.rolling(24).sum()
    df[f"amount_4 h"] = series.rolling(48).sum()
    df[f"amount_6 h"] = series.rolling(12*6).sum()
    df[f"amount_12 h"] = series.rolling(12*12).sum()
    df[f"amount_18 h"] = series.rolling(12*18).sum()
    df[f"amount_24 h"] = series.rolling(12*24).sum()
    df[f"amount_48 h"] = series.rolling(12*48).sum()
    df[f"amount_72 h"] = series.rolling(12*72).sum()

    for wdkzeit in list(KOSTRA_DS.columns):
        for duration_level in list(KOSTRA_DS.index):
            df[f"isHeavyRain_{duration_level}"] = df[f"amount_{duration_level}"] >= KOSTRA_DS[wdkzeit][duration_level]
        duration_level_cols = [f"isHeavyRain_{cur_duration_level}" for cur_duration_level in list(KOSTRA_DS.index)]
        df[f"isHeavyRain_AtLeast{wdkzeit}"] = df[duration_level_cols].max(axis = 1)#sum(axis = 1)
    df = df.drop(columns = duration_level_cols)
    df = df.drop(columns = [f"amount_15 min",
                                f"amount_30 min",
                                f"amount_60 min",
                                f"amount_90 min",
                                f"amount_2 h",
                                f"amount_4 h",
                                f"amount_6 h",
                                f"amount_12 h",
                                f"amount_18 h",
                                f"amount_24 h",
                                f"amount_48 h",
                                f"amount_72 h"
                               ])

    return df.drop(columns = [series.name])

