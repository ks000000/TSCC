from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import random
import re
import os
from .model_error_metrics import *
from ..preprocessing.general import *
from ..preprocessing.plausibility_classes import *
from ..detection.baseline import *
from ..detection.basics import *
from ..detection.statistical import *
from ..detection.ml import *
from ..correction.baseline import *
from ..correction.statistical import *
from ..correction.ml import *

def sample_weights_denseweight(df_fea_list, df_tar_list, detMLConfig):
    dw = DenseWeight(alpha=1.0)
    data_losses_train = dw.fit(train_part["value_plaus"].to_numpy(dtype="float"))

    return sample_weight_train_list

def sample_weights_root(df_fea_list, df_tar_list, detMLConfig):
    data_losses_train = relevance_fct_root(train_part["value_plaus"])

    return sample_weight_train_list

def is_none_or_nan(tp_ind):
    # Check if tp_ind is None
    if tp_ind is None:
        return True
    # Check if tp_ind is NaN
    #if np.issubdtype(tp_ind, np.number) and np.isnan(tp_ind):
    if isinstance(tp_ind, np.number) and np.isnan(tp_ind):
        return True
    return False

def add_eval_of_model(id, model_name, data_set_name, cv, conf_matrx, cm_ind = [None, None, None, None]):

    tn, fp, fn, tp = conf_matrx
    tn_ind, fp_ind, fn_ind, tp_ind = cm_ind
    new_row = {"ID": id,
                    "model": model_name,
                    "data_set": data_set_name,
                    "cv": cv,
                    "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                    "tpr": (np.float64(tp) / (tp + fn)) if (tp + fn) > 0 else np.nan,
                    "tnr": (np.float64(tn) / (tn + fp)) if (tp + fn) > 0 else np.nan,
                    "fpr": (np.float64(fp) / (tn + fp)) if (tp + fn) > 0 else np.nan,
                    "fnr": (np.float64(fn) / (tp + fn)) if (tp + fn) > 0 else np.nan,
                    "recall": (np.float64(tp) / (tp + fn)) if (tp + fn) > 0 else np.nan,
                    "precision": (np.float64(tp) / (tp + fp)) if (tp + fn) > 0 else np.nan,
                    "f-score": (
                                    2 * (np.float64(tp) / (tp + fp)) * (np.float64(tp) / (tp + fn)) /
                                    (
                                            (np.float64(tp) / (tp + fp)) + (np.float64(tp) / (tp + fn))
                                    )
                                ) if (tp + fp) > 0 and (tp + fn) > 0else np.nan,
                    "n_obs": tn+fp+fn+tp
                }

    # add row to model_eval_df of detMLConfig class instance
    return new_row


def add_errclass_eval_of_model(df, target_pred, models_name, data_set_name, detMLConfig, evaluationHandler):

    df["TN"] = (df[detMLConfig.colname_target_det] == False) & (df[target_pred] == False)
    df["FP"] = (df[detMLConfig.colname_target_det] == False) & (df[target_pred] == True)
    df["FN"] = (df[detMLConfig.colname_target_det] == True) & (df[target_pred] == False)
    df["TP"] = (df[detMLConfig.colname_target_det] == True) & (df[target_pred] == True)
    df_errorclass_gr = df[['error_class', 'TN', 'FP', 'FN', 'TP']].groupby("error_class").sum().reset_index()

    for cur_err_class in list(df["error_class"].unique()):
        new_row = {"ID": detMLConfig.colname_raw,
                        "model": models_name,
                        "data_set": data_set_name,
                        "errorclass": cur_err_class,
                        "tn": df_errorclass_gr[df_errorclass_gr["error_class"]==cur_err_class]["TN"].sum(),
                        "fp": df_errorclass_gr[df_errorclass_gr["error_class"]==cur_err_class]["FP"].sum(),
                        "fn": df_errorclass_gr[df_errorclass_gr["error_class"]==cur_err_class]["FN"].sum(),
                        "tp": df_errorclass_gr[df_errorclass_gr["error_class"]==cur_err_class]["TP"].sum(),
                       }
        # add row to model_errorclass_eval of detMLConfig class instance
        evaluationHandler.append_row_to_det_errorclass_eval(new_row)

def get_conf_matrix(y_true, y_pred):
    tn = sum((y_true == False) &
             (y_pred == False))
    fp = sum((y_true == False) &
             (y_pred == True))
    fn = sum((y_true == True) &
             (y_pred == False))
    tp = sum((y_true == True) &
             (y_pred == True))
    conf_matrx = tn, fp, fn, tp
    return conf_matrx

def getEvaluation(df, target_pred, model_name, data_set_name, Config, evaluationHandler, cv ="_"):

    # compute confusion matrix values
    #try:
    #    tn, fp, fn, tp = confusion_matrix(
    #        df[[detMLConfig.colname_target_det]],
    #        df[target_pred]).ravel()
    #except:

    if Config.target_sensor_uncertainty:
        target_det = (abs(df[Config.colname_raw] - df[Config.colname_target_corr]) < Config.target_sensor_uncertainty)
    else:
        target_det = df[Config.colname_target_det]
    conf_matrx = get_conf_matrix(target_det, df[target_pred])

    # compute confusion matrix values for indicator function, value-based
    if np.issubdtype(df[Config.colname_target_corr].dtype, np.number):
        if Config.target_sensor_uncertainty:
            target_det = (abs(df[Config.colname_raw] - df[Config.colname_target_corr]) < Config.target_sensor_uncertainty)
        else:
            target_det = df[Config.colname_target_det]
        relevance_series = relevance_fct_indicator(df[Config.colname_target_corr].reset_index(drop=True), (1.25, float("inf")))
        tn = sum(((target_det == False) &
                  (df[target_pred] == False)).astype(int).reset_index(drop=True)*relevance_series)
        fp = sum(((target_det == False) &
                  (df[target_pred] == True)).astype(int).reset_index(drop=True)*relevance_series)
        fn = sum(((target_det == True) &
                  (df[target_pred] == False)).astype(int).reset_index(drop=True)*relevance_series)
        tp = sum(((target_det == True) &
                  (df[target_pred] == True)).astype(int).reset_index(drop=True)*relevance_series)
        conf_matrx_ind = [tn, fp, fn, tp]
    else:
        conf_matrx_ind = [None, None, None, None]

    # compute confusion matrix values event-based
    if Config.colname_isEvent:
        if Config.target_sensor_uncertainty:
            target_det = (abs(df[Config.colname_raw] - df[Config.colname_target_corr]) < Config.target_sensor_uncertainty)
        else:
            target_det = df[Config.colname_target_det]
        relevance_series = (df[Config.colname_isEvent] == True).astype(int).reset_index(drop=True)
        tn = sum(((target_det == False) &
                  (df[target_pred] == False)).astype(int).reset_index(drop=True)*relevance_series)
        fp = sum(((target_det == False) &
                  (df[target_pred] == True)).astype(int).reset_index(drop=True)*relevance_series)
        fn = sum(((target_det == True) &
                  (df[target_pred] == False)).astype(int).reset_index(drop=True)*relevance_series)
        tp = sum(((target_det == True) &
                  (df[target_pred] == True)).astype(int).reset_index(drop=True)*relevance_series)
        conf_matrx_ev = [tn, fp, fn, tp]

    '''
    if Config.target_sensor_uncertainty:
        gt_unc = (abs(df[Config.colname_raw] - df[Config.colname_target_corr]) < Config.target_sensor_uncertainty)
        tn = sum(((gt_unc == False) &
                  (df[target_pred] == False)).astype(int).reset_index(drop=True))
        fp = sum(((gt_unc == False) &
                  (df[target_pred] == True)).astype(int).reset_index(drop=True))
        fn = sum(((gt_unc == True) &
                  (df[target_pred] == False)).astype(int).reset_index(drop=True))
        tp = sum(((gt_unc == True) &
                  (df[target_pred] == True)).astype(int).reset_index(drop=True))
        conf_matrx_unc = [tn, fp, fn, tp]
    '''

    # add evaluation line to df
    new_row = add_eval_of_model(Config.colname_raw, model_name, data_set_name, cv, conf_matrx, conf_matrx_ind)
    evaluationHandler.append_row_to_det_eval(new_row)
    add_errclass_eval_of_model(df, target_pred, model_name, data_set_name, Config, evaluationHandler)

    # compute confusion matrix values for indicator function, value-based
    if np.issubdtype(df[Config.colname_target_corr].dtype, np.number):
        new_row = add_eval_of_model(Config.colname_raw, model_name, f"r-1.25 {data_set_name}", cv, conf_matrx_ind)
        evaluationHandler.append_row_to_det_eval(new_row)

    # compute confusion matrix values event-based
    if Config.colname_isEvent:
        new_row = add_eval_of_model(Config.colname_raw, model_name, f"isEvent {data_set_name}", cv, conf_matrx_ev)
        evaluationHandler.append_row_to_det_eval(new_row)

    '''
    if Config.target_sensor_uncertainty:
        new_row = add_eval_of_model(Config.colname_raw, model_name, f"uncertainty {data_set_name}", cv, conf_matrx_unc)
        evaluationHandler.append_row_to_det_eval(new_row)
    '''


class EvaluationHandler:
    r"""

    The EvaluationHandler class manages the evaluation of machine learning models, specifically handling metrics for
    both classification and regression tasks. It includes functionality to append new rows of evaluation data,
    and retrieve stored evaluations. It also supports cross-validation and train-test evaluations.

    Parameters
    ----------
    Config : object
        Configuration object that defines model settings for evaluation.
    dataSetHandler : object
        Object responsible for managing the dataset during evaluation.

    Attributes
    ----------
    COLUMN_LIST_EVAL_CAT : list
        List of columns for storing categorical evaluation metrics (e.g., tn, fp, fn, tp, recall, precision).
    COLUMN_LIST_ERRCLASS_EVAL_CAT : list
        List of columns for storing evaluation results based on error classes (e.g., tnr).
    COLUMN_LIST_EVAL_REG : list
        List of columns for storing regression evaluation metrics (e.g., mse, rmse, mae, nse).
    det_eval : pandas DataFrame
        DataFrame that stores categorical evaluation results.
    det_errorclass_eval : pandas DataFrame
        DataFrame that stores evaluation results based on error classes.
    corr_eval : pandas DataFrame
        DataFrame that stores regression evaluation results.

    Methods
    -------
    get_det_eval()
        Retrieves the current categorical evaluation DataFrame (det_eval).
    get_det_errorclass_eval()
        Retrieves the current error class evaluation DataFrame (det_errorclass_eval).
    append_row_to_det_eval(new_row)
        Appends a new row of categorical evaluation data to the det_eval DataFrame.
    append_row_to_det_errorclass_eval(new_row)
        Appends a new row of error class evaluation data to the det_errorclass_eval DataFrame.
    append_row_to_corr_eval(new_row)
        Appends a new row of regression evaluation data to the corr_eval DataFrame.
    """

    # Class constant
    COLUMN_LIST_EVAL_CAT = [
        "ID", "model", "data_set", "cv",
        "tn", "fp", "fn", "tp",
        "tpr", "tnr", "fpr", "fnr",
        # "tpr_ind1.25",
        # "fpr-r-1.25", "fnr-r-1.25",
        "recall", "precision", "f-score",
        "n_obs"]
    COLUMN_LIST_ERRCLASS_EVAL_CAT = ["ID", "model", "data_set", "errorclass", "tnr"]

    COLUMN_LIST_EVAL_REG = ["ID", "model", "data_set", "cv", "mse", "mse-rel-root", "mse-rel-ind 1.25",
                            "ind1-rel-ind 1.25",
                            "rate-rel-ind 1.25", "rmse", "mae", "nse", "n_obs",
                            "sera"]

    def __init__(self, Config, dataSetHandler):
        self.COLUMN_LIST_EVAL_CAT = EvaluationHandler.COLUMN_LIST_EVAL_CAT
        self.COLUMN_LIST_EVAL_REG = EvaluationHandler.COLUMN_LIST_EVAL_REG
        self.det_eval = pd.DataFrame(columns=EvaluationHandler.COLUMN_LIST_EVAL_CAT)
        self.det_errorclass_eval = pd.DataFrame(columns=EvaluationHandler.COLUMN_LIST_ERRCLASS_EVAL_CAT)
        self.corr_eval = pd.DataFrame(columns=EvaluationHandler.COLUMN_LIST_EVAL_REG)

    @staticmethod
    def __drop_empty_or_all_na_columns(df):
        # Drop columns where all values are NaN
        df_cleaned = df.dropna(axis=1, how='all')
        # Optionally, drop columns where there are no non-NA values
        df_cleaned = df_cleaned.loc[:, df_cleaned.notna().any(axis=0)]
        return df_cleaned

    def get_det_eval(self):
        """
        Return the current DataFrame.
        """

        return self.det_eval

    def get_det_errorclass_eval(self):
        """
        Return the current DataFrame.
        """

        return self.det_errorclass_eval

    def append_row_to_det_eval(self, new_row):
        """
        Append a new row to the DataFrame.
        The new row should be in a format that can be converted to a DataFrame.
        """

        new_row_df = pd.DataFrame([new_row])
        new_row_df = EvaluationHandler.__drop_empty_or_all_na_columns(new_row_df)
        if not self.det_eval.empty:
            self.det_eval = pd.concat([self.det_eval, new_row_df], ignore_index=True)
        else:
            self.det_eval = new_row_df

    def append_row_to_det_errorclass_eval(self, new_row):
        """
        Append a new row to the DataFrame.
        The new row should be in a format that can be converted to a DataFrame.
        """

        new_row_df = pd.DataFrame([new_row])
        new_row_df = EvaluationHandler.__drop_empty_or_all_na_columns(new_row_df)
        if not self.det_errorclass_eval.empty:
            self.det_errorclass_eval = pd.concat([self.det_errorclass_eval, new_row_df], ignore_index=True)
        else:
            self.det_errorclass_eval = new_row_df

    def append_row_to_corr_eval(self, new_row):
        """
        Append a new row to the DataFrame.
        The new row should be in a format that can be converted to a DataFrame.
        """

        new_row_df = pd.DataFrame([new_row])
        #new_row_df = EvaluationHandler.__drop_empty_or_all_na_columns(new_row_df)
        if not self.corr_eval.empty:
            self.corr_eval = pd.concat([self.corr_eval, new_row_df], ignore_index=True)
        else:
            self.corr_eval = new_row_df

def det_evaluate_method(func, dataSetHandler, config, *args):
    d_part = {}
    for i in range(dataSetHandler.list_len):
        d_part[i] = {}

    for i in range(dataSetHandler.list_len):
        if "STAT" in f"{func}" or "BASIC" in f"{func}":
            d_part[i]["train"] = func(
                dataSetHandler.get_train_features()[i][config.colname_raw].copy(deep=True),
                #None,
                #detMLConfig,
                *args)
            d_part[i]["test"] = func(
                dataSetHandler.get_test_features()[i][config.colname_raw].copy(deep=True),
                #None,
                #detMLConfig,
                *args)
        elif "ML" in f"{func}":
            class_instance = func()
            class_instance.fit(
                dataSetHandler.get_train_features(exclude_columns=config.exclude_cols)[i].copy(deep=True),
                dataSetHandler.get_train_targets(extract_single_target=config.colname_target_det)[i].copy(
                    deep=True),
                config,
                *args)
            d_part[i]["train"] = class_instance.predict(
                dataSetHandler.get_train_features(exclude_columns=config.exclude_cols)[i].copy(deep=True))
            d_part[i]["test"] = class_instance.predict(
                dataSetHandler.get_test_features(exclude_columns=config.exclude_cols)[i].copy(deep=True))
            if config.det_ML_model_savefolder:
                match = re.search(r"<class '[^']*\.(\w+)'>", f"{func}")
                if match:
                    short_funcname = match.group(1)
                else:
                    short_funcname = "model"
                if not os.path.exists(f"{config.det_ML_model_savefolder}\\{config.colname_raw}"):
                    os.makedirs(f"{config.det_ML_model_savefolder}\\{config.colname_raw}")
                class_instance.save_model(f"{config.det_ML_model_savefolder}\\{config.colname_raw}\\{short_funcname}_cv{i}.pkl")
                data = {"train_event_absnr": [dataSetHandler.train_event_absnr[i]],
                        "train_event_rel": [dataSetHandler.train_eventrates[i]],
                        "train_timerange": [f"{dataSetHandler.get_train_features()[i].index.min(), dataSetHandler.get_train_features()[i].index.max()}"]

                        }
                pd.DataFrame(data).to_csv(f"{config.det_ML_model_savefolder}\\{config.colname_raw}\\additional_info_cv{i}.csv", index=False)
        elif "BASE" in f"{func}":
            d_part[i]["train"] = func(
                dataSetHandler.get_train_features()[i].copy(deep=True),
                dataSetHandler.get_train_targets()[i].copy(deep=True),
                config,
                *args)
            d_part[i]["test"] = func(
                dataSetHandler.get_test_features()[i].copy(deep=True),
                dataSetHandler.get_test_targets()[i].copy(deep=True),
                config,
                *args)
        else:
            print(f"Error in data and function handler regarding {func}.")
            pass

    return d_part

def extract_functions_and_parameters(methods):
    functions = []
    parameters = []
    for cur_meth in methods:
        #if multiple functions and/ or additional parameters are given
        if isinstance(cur_meth, tuple):
            # multiple functions are given
            if isinstance(cur_meth[0], tuple):
                # Handle nested tuple
                functions_chain = []
                parameters_chain = []
                for nested_meth in cur_meth:
                    functions_chain.append(nested_meth[0])
                    parameters_chain.append(nested_meth[1:])
                functions.append(functions_chain)
                parameters.append(parameters_chain)
            # additional parameters are given
            else:
                functions.append(cur_meth[0])
                parameters.append(cur_meth[1:])
        else:
            functions.append(str(cur_meth))
            parameters.append(())

    return functions, parameters


def detection_evaluation_wrapper(config,
                                 dataSetHandler,
                                 det_ML_methods=[],
                                 det_stat_methods=[],
                                 ):
    r"""
    Wraps error detection modeling and evaluation, applying both machine learning and statistical methods
    for anomaly detection. This function manages the execution of models, performs cross-validation if configured,
    and stores the results in a nested dictionary structure and a class instance of the evaluationHandler.

    Parameters
    ----------
    config : class instance of Config
        A mutable class instance containing the configuration settings for model evaluation, such as
        column names, cross-validation settings, and other options.
    dataSetHandler : class instance of DataSetHandler
        An instance responsible for storing and handling the dataset. It manages both the features
        and targets for training and testing data.
    det_ML_methods : list, optional
        A list of machine learning methods for detection (default is an empty list). These methods
        will be applied for error detection modeling.
    det_stat_methods : list, optional
        A list of statistical methods for detection (default is an empty list). These methods will
        be applied for error detection modeling.

    Returns
    -------
    d : dict
        A nested dictionary, where each key corresponds to a machine learning or statistical method.
        The value is a sub-dictionary that contains the train and test predictions (as pandas Series)
        and the generated machine learning models.
    evaluationHandler : object
        A list containing the generated machine learning models for each method used in the detection.
    """

    # initialization of returned parameters
    d = {}
    evaluationHandler = EvaluationHandler(config, dataSetHandler)

    meth_functions, meth_parameters = extract_functions_and_parameters(det_stat_methods + det_ML_methods)
    for i, cur_method in enumerate(meth_functions):
        run_len = len(cur_method) if isinstance(cur_method, list) else 1
        for j in range(run_len):
            # if ground truth is numerical
            cur_method_unchained = cur_method[j] if isinstance(cur_method, list) else cur_method
            cur_meth_param_unchained = meth_parameters[i][j] if isinstance(meth_parameters[i], list) else meth_parameters[i]
            if np.issubdtype(dataSetHandler.get_train_targets()[0][config.colname_target_corr].dtype, np.number):
                if len(meth_parameters[i])> 0:
                    d[str(cur_method)] = det_evaluate_method(
                                    cur_method_unchained,
                                    dataSetHandler,
                                    config,
                                    *cur_meth_param_unchained)
                else:
                    d[str(cur_method)] = det_evaluate_method(
                        cur_method_unchained,
                        dataSetHandler,
                        config)

            # if more than one run is executed
            if run_len > 1:
                if j > 0:
                    for cur_cv_key in d[str(cur_method)].keys():
                        d[str(cur_method)][cur_cv_key]["train"] = np.maximum(d[str(cur_method)][cur_cv_key]["train"], int_results[cur_cv_key]["train"])
                        int_results[cur_cv_key]["train"] = np.maximum(d[str(cur_method)][cur_cv_key]["train"], int_results[cur_cv_key]["train"])
                        d[str(cur_method)][cur_cv_key]["test"] = np.maximum(d[str(cur_method)][cur_cv_key]["test"], int_results[cur_cv_key]["test"])
                        int_results[cur_cv_key]["test"] = np.maximum(d[str(cur_method)][cur_cv_key]["test"], int_results[cur_cv_key]["test"])
                if j == 0:
                    int_results = d[str(cur_method)]

    # evaluate models
    for cur_ml_fct in d.keys():
        for cur_cv_set in d[cur_ml_fct].keys():
            if cur_cv_set == 'ml_model':
                continue

            train_data = {
                f"{config.colname_raw}": dataSetHandler.get_train_features()[int(cur_cv_set)][config.colname_raw],
                f"{config.colname_target_det}": dataSetHandler.get_train_targets()[int(cur_cv_set)][config.colname_target_det],
                #f"hasGoodDQ_{cur_ml_fct}": dataSetHandler.get_train_targets()[int(cur_cv_set)][detMLConfig.colname_target_det],
                f"hasGoodDQ_{cur_ml_fct}": pd.Series(d[cur_ml_fct][int(cur_cv_set)]["train"]),
                "error_class": dataSetHandler.get_train_features()[int(cur_cv_set)]["error_class"],
                f"{config.colname_isEvent}": dataSetHandler.get_train_features()[int(cur_cv_set)][f"{config.colname_isEvent}"],
                f"{config.colname_target_corr}": dataSetHandler.get_train_targets()[int(cur_cv_set)][f"{config.colname_target_corr}"]
            }
            getEvaluation(
                df=pd.DataFrame(train_data),
                target_pred=f"hasGoodDQ_{cur_ml_fct}",
                model_name = f"{cur_ml_fct}",
                data_set_name = "train",
                Config= config,
                evaluationHandler= evaluationHandler,
                cv = cur_cv_set)

            test_data = {
                f"{config.colname_raw}": dataSetHandler.get_test_features()[int(cur_cv_set)][config.colname_raw],
                f"{config.colname_target_det}": dataSetHandler.get_test_targets()[int(cur_cv_set)][config.colname_target_det],
                #f"hasGoodDQ_{cur_ml_fct}": dataSetHandler.get_test_targets()[int(cur_cv_set)][detMLConfig.colname_target_det],
                f"hasGoodDQ_{cur_ml_fct}": pd.Series(d[cur_ml_fct][int(cur_cv_set)]["test"]),
                "error_class": dataSetHandler.get_test_features()[int(cur_cv_set)]["error_class"],
                f"{config.colname_isEvent}": dataSetHandler.get_test_features()[int(cur_cv_set)][f"{config.colname_isEvent}"],
                f"{config.colname_target_corr}": dataSetHandler.get_test_targets()[int(cur_cv_set)][f"{config.colname_target_corr}"]
            }
            getEvaluation(
                df=pd.DataFrame(test_data),
                target_pred=f"hasGoodDQ_{cur_ml_fct}",
                model_name = f"{cur_ml_fct}",
                data_set_name = "test",
                Config= config,
                evaluationHandler= evaluationHandler,
                cv = cur_cv_set)

    d.update({'isCV': config.cross_validation})
    d.update({'train_test_IDs': config.train_test_IDs})

    return d, evaluationHandler

def feature_generation(df):
    # Wrapping feature generation
    # feature - generaation    function as parameter(usage    of    different    features)?

    #    df_fea = feature_generation(df)
    return df

def wrapper_model_application(ml_fct_list, df):
    # hier: Nutzen von gespeicherten Modellen
    #

    return pd.Series()


def add_eval_of_model_reg(config, model_name, data_set_name, cv, y_gt, y_pred):
    # drop na's
    df = pd.DataFrame([y_gt, y_pred], index=["y_gt", "y_pred"]).transpose().dropna()
    y_gt = df["y_gt"]
    y_pred = df["y_pred"]

    new_row = {
        "ID": config.colname_raw,
        "model": model_name,
        "data_set": data_set_name,
        "cv": cv,
        "mse": mean_squared_error(y_gt, y_pred, squared=True) if not y_gt.empty else np.nan,
        "mse-rel-root": mean_squared_error(y_gt, y_pred, sample_weight = relevance_fct_root(y_gt)) if sum(relevance_fct_root(y_gt))> 0 else np.nan,
        "mse-rel-ind 1.25": mean_squared_error(y_gt, y_pred, sample_weight = relevance_fct_indicator(y_gt, (1.25, float("inf")))) if sum(relevance_fct_indicator(y_gt, (1.25, float("inf"))))> 0 else np.nan,
        "ind1-rel-ind 1.25": indicator_error_relevance(y_gt, y_pred, (0, 1), relevance_fct_indicator, (1.25,float("inf"))),
        "rate-rel-ind 1.25": rate_error_relevance(y_gt, y_pred, relevance_fct_indicator, (1.25,float("inf"))),
        "rmse": mean_squared_error(y_gt, y_pred, squared=False) if not y_gt.empty else np.nan,
        "mae": mean_absolute_error(y_gt, y_pred) if not y_gt.empty else np.nan,
        "nse": NSE(y_pred, y_gt),
        "n_obs": len(y_gt),
        #"abs_vol_error": abs_vol_error(y_pred, y_gt),
        #"rel_vol_error": rel_vol_error(y_pred, y_gt),
        "sera": squared_error_relevance_area(y_gt, y_pred, relevance_fct_root(y_gt))[0]
       }

    # add row to model_errorclass_eval of detMLConfig class instance
    return new_row


def getEvaluation_reg(df, colname_pred, model_name, data_set_name, config, evaluationHandler, cv ="-"):

    #all data
    new_row = add_eval_of_model_reg(config, model_name, data_set_name, cv, df[config.colname_target_corr], df[colname_pred])
    evaluationHandler.append_row_to_corr_eval(new_row)

    # compute confusion matrix values for indicator function, value-based
    if np.issubdtype(df[config.colname_target_corr].dtype, np.number):
        relevance_series = relevance_fct_indicator(df[config.colname_target_corr], (1.25, float("inf")))
        new_row = add_eval_of_model_reg(config, model_name, f"r-1.25 {data_set_name}", cv,
                              df[config.colname_target_corr][relevance_series > 0.0], df[colname_pred][relevance_series > 0.0])
        evaluationHandler.append_row_to_corr_eval(new_row)
    # compute event-based
    if config.colname_isEvent:
        relevance_series = df[config.colname_isEvent].astype(int)#.reset_index(drop=True)
        new_row = add_eval_of_model_reg(config, model_name, f"isEvent {data_set_name}", cv,
                              df[config.colname_target_corr][relevance_series > 0.0], df[colname_pred][relevance_series > 0.0])
        evaluationHandler.append_row_to_corr_eval(new_row)



def isIncorrect_evaluate(series_isCorrect, series_raw, series_target):
    return (series_isCorrect == False) | (series_raw != series_target)


def corr_evaluate_method(func, dataSetHandler, config, dataSetHandlerColumn_errorDetection, *args):
    d_part = {}
    for i in range(dataSetHandler.list_len):
        d_part[i] = {}

    for i in range(dataSetHandler.list_len):
        if "STAT" in f"{func}":
            d_part[i]["train"] = func(
                dataSetHandler.get_train_features()[i].copy(deep=True),#[config.colname_raw]
                None,
                dataSetHandlerColumn_errorDetection,
                config,
                *args)
            d_part[i]["test"] = func(
                dataSetHandler.get_test_features()[i].copy(deep=True),
                None,
                dataSetHandlerColumn_errorDetection,
                config,
                *args)
        elif "ML" in f"{func}":
            class_instance = func()
            class_instance.fit(
                dataSetHandler.get_train_features(exclude_columns=config.exclude_cols)[i].copy(deep=True),
                dataSetHandler.get_train_targets(extract_single_target=config.colname_target_corr)[i].copy(
                    deep=True),
                dataSetHandlerColumn_errorDetection,
                config,
                *args)
            d_part[i]["train"] = class_instance.predict(
                dataSetHandler.get_train_features(exclude_columns=config.exclude_cols)[i].copy(deep=True),
                dataSetHandlerColumn_errorDetection,
                config)
            d_part[i]["test"] = class_instance.predict(
                dataSetHandler.get_test_features(exclude_columns=config.exclude_cols)[i].copy(deep=True),
                dataSetHandlerColumn_errorDetection,
                config)
            if config.corr_ML_model_savefolder:
                match = re.search(r"<class '[^']*\.(\w+)'>", f"{func}")
                if match:
                    short_funcname = match.group(1)
                else:
                    short_funcname = "model"
                if not os.path.exists(f"{config.corr_ML_model_savefolder}\\{config.colname_raw}"):
                    os.makedirs(f"{config.corr_ML_model_savefolder}\\{config.colname_raw}")
                class_instance.save_model(f"{config.corr_ML_model_savefolder}\\{config.colname_raw}\\{short_funcname}_cv{i}.pkl")
                data = {"train_event_absnr": [dataSetHandler.train_event_absnr[i]],
                        "train_event_rel": [dataSetHandler.train_eventrates[i]],
                        "train_timerange": [f"{dataSetHandler.get_train_features()[i].index.min(), dataSetHandler.get_train_features()[i].index.max()}"]

                        }
                pd.DataFrame(data).to_csv(f"{config.corr_ML_model_savefolder}\\{config.colname_raw}\\additional_info_cv{i}.csv", index=False)
        elif "BASE" in f"{func}":
            d_part[i]["train"] = func(
                dataSetHandler.get_train_features()[i].copy(deep=True),
                dataSetHandler.get_train_targets()[i].copy(deep=True),
                config,
                *args)
            d_part[i]["test"] = func(
                dataSetHandler.get_test_features()[i].copy(deep=True),
                dataSetHandler.get_test_targets()[i].copy(deep=True),
                config,
                *args)
        else:
            print(f"{func} not available.")
            pass

    return d_part

def correction_evaluation_wrapper(config,
                                  dataSetHandler,
                                  dataSetHandlerColumn_errorDetection,
                                  corr_ML_methods=[],
                                  corr_stat_methods=[],
                                  ):
    r"""
    Wraps error correction modeling and evaluation, applying machine learning and statistical methods for
    correcting detected errors. This function manages the application of correction models, performs cross-validation if
    configured, and stores the results for further evaluation.

    Parameters
    ----------
    config : class instance of Config
        A mutable class instance that contains configuration settings for correction models, such as column names,
        cross-validation flags, and other relevant options.
    dataSetHandler : class instance of DataSetHandler
        An instance responsible for storing and handling the dataset, managing both the feature and target data
        for training and testing.
    dataSetHandlerColumn_errorDetection : str
        The column in the dataset that marks whether an error has been detected. This column is used to identify
        erroneous entries for correction.
    corr_ML_methods : list, optional
        A list of machine learning methods to be applied for error correction modeling (default is an empty list).
    corr_stat_methods : list, optional
        A list of statistical methods to be applied for error correction modeling (default is an empty list).

    Returns
    -------
    d : dict
        A nested dictionary where each key corresponds to a correction method (either machine learning or statistical).
        The value is a sub-dictionary that contains the train and test predictions, the corrected target values,
        and other evaluation results.
    evaluationHandler : EvaluationHandler
        An instance of `EvaluationHandler` that contains the evaluation results for the applied correction models.

    Notes
    -----
    - For each correction method, the results from multiple runs (if applicable) are combined using averaging for
      both train and test datasets.

    """

    # initialization of returned parameters
    d = {}
    evaluationHandler = EvaluationHandler(config, dataSetHandler)
    '''
    for cur_method in detMLConfig.corr_ML_methods + detMLConfig.corr_stat_methods:
        cur_method = cur_method  # f"{cur_method}".split(" ")[1]
        d[cur_method] = {}
        for i in range(dataSetHandler.list_len):
            d[cur_method][i] = {}
    '''

    d["add_info"] = {}
    pred_dict = {}
    for i in range(dataSetHandler.list_len):
        d["add_info"][i] = 1

    #meth_functions = [cur_meth[0] if isinstance(cur_meth, tuple) else str(cur_meth) for cur_meth in
    #                  config.corr_stat_methods + config.corr_ML_methods]
    #meth_parameters = [cur_meth[1:] if isinstance(cur_meth, tuple) else () for cur_meth in
    #                   config.corr_stat_methods + config.corr_ML_methods]
    meth_functions, meth_parameters = extract_functions_and_parameters(corr_stat_methods + corr_ML_methods)
    for i, cur_method in enumerate(meth_functions):
        run_len = len(cur_method) if isinstance(cur_method, list) else 1
        for j in range(run_len):
            cur_method_unchained = cur_method[j] if isinstance(cur_method, list) else cur_method
            cur_meth_param_unchained = meth_parameters[i][j] if isinstance(meth_parameters[i], list) else \
                meth_parameters[i]
            if len(meth_parameters[i]) > 0:
                d[str(cur_method)] = corr_evaluate_method(
                    cur_method_unchained,
                    dataSetHandler,
                    config,
                    dataSetHandlerColumn_errorDetection,
                    *cur_meth_param_unchained)
            else:
                d[str(cur_method)] = corr_evaluate_method(
                    cur_method_unchained,
                    dataSetHandler,
                    config,
                    dataSetHandlerColumn_errorDetection)

            # if more than one run is executed
            if run_len > 1:
                if j > 0:
                    for cur_cv_key in d[str(cur_method)].keys():
                        d[str(cur_method)][cur_cv_key]["train"] = pd.concat([d[str(cur_method)][cur_cv_key]["train"], int_results[cur_cv_key]["train"]], axis = 1).mean(axis = 1)
                        int_results[cur_cv_key]["train"] = pd.concat([d[str(cur_method)][cur_cv_key]["train"], int_results[cur_cv_key]["train"]], axis = 1).mean(axis = 1)
                        d[str(cur_method)][cur_cv_key]["test"] = pd.concat([d[str(cur_method)][cur_cv_key]["test"], int_results[cur_cv_key]["test"]], axis = 1).mean(axis = 1)
                        int_results[cur_cv_key]["test"] = pd.concat([d[str(cur_method)][cur_cv_key]["test"], int_results[cur_cv_key]["test"]], axis = 1).mean(axis = 1)
                if j == 0:
                    int_results = d[str(cur_method)]


    # evaluate models
    for cur_ml_fct in d.keys():
        if cur_ml_fct in ["add_info"]:
            continue
        for cur_cv_set in d[cur_ml_fct].keys():
            if cur_cv_set in ['ml_model']:
                continue
            print(cur_ml_fct)

            evaluate_train = isIncorrect_evaluate(series_isCorrect = ~dataSetHandler.get_train_features()[int(cur_cv_set)][dataSetHandlerColumn_errorDetection],
                                                  series_raw = dataSetHandler.get_train_features()[int(cur_cv_set)][config.colname_raw],
                                                  series_target = dataSetHandler.get_train_targets()[int(cur_cv_set)][config.colname_target_corr])
            train_data = {
                    f"{config.colname_target_corr}": dataSetHandler.get_train_targets()[int(cur_cv_set)][evaluate_train][config.colname_target_corr],
                    #f"corr_{cur_ml_fct}": dataSetHandler.get_train_targets()[int(cur_cv_set)][evaluate_train][config.colname_target_corr],
                    f"corr_{cur_ml_fct}": pd.Series(d[cur_ml_fct][int(cur_cv_set)]["train"][evaluate_train]),
                    "error_class": dataSetHandler.get_train_features()[int(cur_cv_set)][evaluate_train]["error_class"],
                    f"{config.colname_isEvent}": dataSetHandler.get_train_features()[int(cur_cv_set)][evaluate_train][config.colname_isEvent],
                }
            getEvaluation_reg(
                df=pd.DataFrame(train_data),
                colname_pred=f"corr_{cur_ml_fct}",
                model_name=f"{cur_ml_fct}",
                data_set_name="train",
                config=config,
                evaluationHandler=evaluationHandler,
                cv=cur_cv_set)

            evaluate_test = isIncorrect_evaluate(
                series_isCorrect=~dataSetHandler.get_test_features()[int(cur_cv_set)][dataSetHandlerColumn_errorDetection],
                series_raw=dataSetHandler.get_test_features()[int(cur_cv_set)][config.colname_raw],
                series_target=dataSetHandler.get_test_targets()[int(cur_cv_set)][config.colname_target_corr])
            test_data = {
                f"{config.colname_target_corr}": dataSetHandler.get_test_targets()[int(cur_cv_set)][evaluate_test][
                    config.colname_target_corr],
                #f"corr_{cur_ml_fct}": dataSetHandler.get_test_targets()[int(cur_cv_set)][evaluate_test][config.colname_target_corr],
                f"corr_{cur_ml_fct}": pd.Series(d[cur_ml_fct][int(cur_cv_set)]["test"][evaluate_test]),
                "error_class": dataSetHandler.get_test_features()[int(cur_cv_set)][evaluate_test]["error_class"],
                f"{config.colname_isEvent}": dataSetHandler.get_test_features()[int(cur_cv_set)][evaluate_test][
                    config.colname_isEvent],
            }
            getEvaluation_reg(
                df=pd.DataFrame(test_data),
                colname_pred=f"corr_{cur_ml_fct}",
                model_name=f"{cur_ml_fct}",
                data_set_name="test",
                config=config,
                evaluationHandler=evaluationHandler,
                cv=cur_cv_set)
            

    d.update({'isCV': config.cross_validation})
    d.update({'train_test_IDs': config.train_test_IDs})

    return d, evaluationHandler








