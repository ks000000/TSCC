import numpy as np
import pandas as pd
from scipy.integrate import simpson

def relevance_fct_root(series, addend = 0.00001):
    r"""
    Root function

    Parameters
    ----------
    series : pandas series
        Pandas series of which entries are assessed regarding its relevance.
    addend : float
        Addend of root function

    Returns
    -------
    series : pandas series
        Pandas series of boolean dtype.
    """
    return np.sqrt(series) + addend

def relevance_fct_indicator(series, subset_boundary = (-float("inf"), float("inf"))):
    r"""
    Indicator function

    Parameters
    ----------
    series : pandas series
        Pandas series of which entries are assessed regarding its relevance.
    subset_boundary : tuple
        Lower and upper boundary of subset receiving value one, and all other elements to zero

    Returns
    -------
    series : pandas series
        Pandas series of boolean dtype.
    """
    return (series >= subset_boundary[0]) & (series <= subset_boundary[1])#pd.Series(np.where(series >= threshold, 1, 0), index=series.index)

'''
def mean_squared_error_relevance(y_gt, y_pred, relevance_function, *extraArgs):
    r"""
    Measure the mean squared error of your ground truth dataset and your prediction.

    Parameters
    ----------
    y_gt : pandas series
        Pandas series of the ground truth dataset.
    y_pred : pandas series
        Pandas series of your prediction.
    relevance_function : function
        A function you can use to set a threshold, so lower gt-values won't be included in this measurement.
    *extraArgs : int or float
        A threshold for relevance_function.

    Returns
    -------
    mse_rel : int or float
        The mean squared error of your two datasets.

    Examples
    --------
    >>> gt_series = pd.Series([17, 99])
    >>> pred_series = pd.Series([80, 49])
    >>> mse_rel = mean_squared_error_relevance(gt_series, pred_series, relevance_fct_indicator, 1)
    >>> mse_rel
    3234.5
    """

    y_gt = y_gt.reset_index(drop=True)
    y_pred = y_pred.reset_index(drop=True)

    y_gt_relevance = relevance_function(y_gt, *extraArgs)#getRelevance(y_gt, relevance_function = relevance_function)
    if sum(y_gt_relevance)==0:
        mse_rel = np.nan
    else:
        mse_rel = sum(pow(y_pred-y_gt, 2)*y_gt_relevance)/sum(y_gt_relevance)#(len(y_gt))
    return mse_rel
'''

def indicator_error_relevance(y_gt, y_pred, subset_boundary, relevance_function, *extraArgs):
    r"""
    Measure the indicator error and relevance of each value.

    Parameters
    ----------
    y_gt : pandas series
        Pandas series of the ground truth dataset.
    y_pred : pandas series
        Pandas series of your prediction.
    threshold : int or float
        A threshold for the difference of gt-value and pred-value. If the difference is lower than the
        threshold, then this values become irrelevant for the indicator.
    relevance_function : function
        A function you can use to set a threshold, so lower gt-values won't be included in this measurement.
    *extraArgs : int or float
        A threshold for relevance_function.

    Returns
    -------
    ind_rel : int or float
        The indicator error.

    Examples
    --------
    >>> gt_series = pd.Series([1, 2])
    >>> pred_series = pd.Series([3, 4])
    >>> ind_rel = indicator_error_relevance(gt_series, pred_series, relevance_fct_indicator, 1)
    >>> ind_rel
    1.0
    """

    y_gt = y_gt.reset_index(drop=True)
    y_pred = y_pred.reset_index(drop=True)

    y_gt_relevance = relevance_function(y_gt, *extraArgs)
    if sum(y_gt_relevance)==0:
        ind_rel = np.nan
    else:
        isInSubset = (abs(y_pred-y_gt) >= subset_boundary[0]) & (abs(y_pred-y_gt) <= subset_boundary[1])
        ind_rel = sum(isInSubset & y_gt_relevance)/sum(y_gt_relevance)#sum(pd.Series(np.where(abs(y_pred-y_gt)>= threshold, 1, 0))*y_gt_relevance)/sum(y_gt_relevance)#(len(y_gt))
    return ind_rel

def rate_error_relevance(y_gt, y_pred, relevance_function, *extraArgs):
    r"""
    Measure the error rate and relevance of each value.

    Parameters
    ----------
    y_gt : pandas series
        Pandas series of the ground truth dataset.
    y_pred : pandas series
        Pandas series of your prediction.
    relevance_function : function
        A function you can use to set a threshold, so lower gt-values won't be included in this measurement.
    *extraArgs : int or float
        A threshold for relevance_function.

    Returns
    -------
    rate_rel : int or float
        The error rate.

    Examples
    --------
    >>> gt_series = pd.Series([1, 2])
    >>> pred_series = pd.Series([3, 4])
    >>> rate_rel = rate_error_relevance(gt_series, pred_series, relevance_fct_indicator, 0)
    >>> rate_rel
    1.5
    """

    y_gt = y_gt.reset_index(drop=True)
    y_pred = y_pred.reset_index(drop=True)

    y_gt_relevance = relevance_function(y_gt, *extraArgs)#getRelevance(y_gt, relevance_function = relevance_function)
    if sum(y_gt_relevance)==0:
        rate_rel = np.nan
    else:
        rate_rel = ((abs(y_gt - y_pred)/y_gt)*y_gt_relevance).sum(skipna=True)/sum(y_gt_relevance)#(len(y_gt))
    return rate_rel

def squared_error_relevance_area(y_gt, y_pred, y_relevance=None, interval_size = 0.001):
    r"""

    Measure the squared error relevance and its area proposed by Ribeiro and Moniz, 2020

    Parameters
    ----------
    y_gt : pandas series
        Pandas series of the ground truth dataset.
    y_pred : pandas series
        Pandas series of your prediction.
    y_relevance : pandas series
        Pandas series of the obervations relevances.
    interval_size : float, optional
        An interval size between 0 and 1. The default is 0.001.

    Returns
    -------
    sera : float
        The squared error relevance area.
    df_ser : pandas dataframe
        The squared error relevances for each phi(y).

    Examples
    --------
    >>> gt_series = pd.Series([1, 2])
    >>> pred_series = pd.Series([3, 4])
    >>> sera, ser = squared_error_relevance_area(gt_series, pred_series, relevance_fct_root(gt_series))
    >>> sera
    1.5
    >>> plt.plot(ser["phi"], ser["ser_phi"])
    """

    # creating a list from 0 to 1 with possible cutoffs t based on interval_size
    start_range = 0
    end_range = 1
    cutoffs = list(np.arange(start_range, end_range, interval_size))
    cutoffs.append(end_range)
    cutoffs = sorted(cutoffs, key=lambda x: float(x))

    df_help = pd.concat([y_gt,y_pred,y_relevance],axis=1,keys= ['true', 'pred', 'phi'])
    # Initiating lists to store squared-error relevance (ser)
    ser_phi = []

    # selecting a phi value
    for phi in cutoffs:
        error_squared_sum_phi = 0
        error_squared_sum_phi = sum((df_help[df_help.phi>=phi]['true'] - df_help[df_help.phi>=phi]['pred'])**2)
        ser_phi.append(error_squared_sum_phi)

    df_ser = pd.DataFrame([cutoffs,ser_phi], index= ['phi', 'ser_phi']).transpose()

    # numerical integration using simpson(y, x)
    sera = simpson(ser_phi, cutoffs)

    return sera, df_ser


def abs_vol_error(y_gt, y_pred):
    r"""
    Measure the absolute volume error (AVE)
    assuming equal weighting (equidistant time steps) for all observations

    Parameters
    ----------
    y_gt : pandas series
        Pandas series of the ground truth dataset.
    y_pred : pandas series
        Pandas series of your prediction.

    Returns
    -------
    ave : float
        The absolute volume error.

    Examples
    --------
    >>> gt_series = pd.Series([1, 2])
    >>> pred_series = pd.Series([3, 4])
    >>> ave = TSCC.assessment.abs_vol_error(gt_series, pred_series)
    >>> ave
    XY
    """

    return (y_pred-y_gt).sum()


#Fkt muss noch abgewandelt werden, für egal welche Einheit des Parameters!!
def rel_vol_error(y_gt, y_pred):
    r"""
    Measure the relative volume error (RVE)
    assuming equal weighting (equidistant time steps) for all observations

    Parameters
    ----------
    y_gt : pandas series
        Pandas series of the ground truth dataset.
    y_pred : pandas series
        Pandas series of your prediction.

    Returns
    -------
    rve : float
        The relative volume error.

    Examples
    --------
    >>> gt_series = pd.Series([1, 2])
    >>> pred_series = pd.Series([3, 4])
    >>> ave = rel_vol_error(gt_series, pred_series)
    >>> rve
    XY
    """

    return (100 * abs_vol_error(y_gt, y_pred) / np.nansum(y_gt.sum()))


def peak_error(y_gt, y_pred):
    r"""
    Measure the peak error
    assuming equal weighting (equidistant time steps) for all observations

    Parameters
    ----------
    y_gt : pandas series
        Pandas series of the ground truth dataset with datetime index.
    y_pred : pandas series
        Pandas series of your prediction with datetime index.

    Returns
    -------
    ape : float
        Absolute Peak Error (APE), absolute deviation from peak value.
    rpe : float
        Relative Peak Error (RPE), relative deviation from peak value.
    pte : float
        Peak Time Error (PTE) , time deviation from peak time.

    Examples
    --------
    >>> gt_series = pd.Series([1, 2])# index must be datetime
    >>> pred_series = pd.Series([3, 4])# index must be datetime
    >>> ape, rpe, pte = peak_error(gt_series, pred_series)
    >>> rve
    XY
    """

    # Get the indices (Timestamps as Datetime) and value of the peak for y_true
    y_gt_max_value = np.nanmax(np.array(y_gt))
    y_gt_max_time = pd.Series(y_gt).idxmax()
    # Get it for y_pred
    y_pred_max_value = np.nanmax(np.array(y_pred))
    y_pred_max_time = pd.Series(y_pred).idxmax()
    # Calculate metrics
    ape = (y_pred_max_value - y_gt_max_value)
    rpe = (100 * (y_pred_max_value - y_gt_max_value) / y_gt_max_value)
    pte = y_pred_max_time - y_gt_max_time

    return ape, rpe, pte

def NSE(y_pred, y_true):
    r"""
    Measure the Nash-Sutcliffe Efficiency (NSE)

    Parameters
    ----------
    y_gt : pandas series
        Pandas series of the ground truth dataset.
    y_pred : pandas series
        Pandas series of your prediction.

    Returns
    -------
    float
        The Nash-Sutcliffe Efficiency.

    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_true_mean = np.nanmean(y_true)

    numerator = np.nansum((y_pred - y_true) ** 2)
    denominator = np.nansum((y_pred - y_true_mean) ** 2)

    return np.round((1 - (numerator / denominator)), decimals=2)

