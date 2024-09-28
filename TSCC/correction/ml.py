import pandas as pd
import numpy as np
from denseweight import DenseWeight
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib

from ..additional_imports.missingpy import missforest
from ..additional_imports.gain import gain


class ML_byGAIN_DW:
    # TODO: extra fit and transform methods missing. Right now only fit_transform part

    """
    This class implements a model for data imputation using the GAIN (Generative Adversarial Imputation Nets) framework
    by https://github.com/jsyoon0823/GAIN.
    The model is designed to fill in missing values in datasets by leveraging a deep learning approach.

    Right now only fit_transform part, therefore also newly trained model for test dataset.

    Parameters
    ----------
    gain_parameters: dict, optional
        A dictionary containing the parameters for the GAIN model, including 'batch_size', 'hint_rate',
        'alpha', and 'iterations'. Default parameters will be used if not provided.
    alpha: float, default=1.0
        The weight for the dense loss function, influencing the training dynamics.
    exclude_cols: list, optional
        A list of columns to exclude from the training process. Not implemented in the current version.
    """
    def __init__(self, gain_parameters=None, alpha=1.0, exclude_cols=None):

        if gain_parameters is None:
            gain_parameters = {
                'batch_size': 128,
                'hint_rate': 0.9,
                'alpha': 100,
                'iterations': 10
            }
        self.gain_parameters = gain_parameters
        self.dw = DenseWeight(alpha=alpha)

    def fit(self, df_fea, df_tar, df_flag_erroneous, config):
        if config.dataset_size == "large":
            # do not run this method, takes very long to compute
            print("Model ML_byGAIN_DW cannot be computed.")

        try:
            data_losses_fit = 1

            # Train Data Imputation
            train_help_df = df_fea.fillna(-7777)
            train_array = train_help_df.to_numpy(dtype="float")
            train_imputed_data = gain(train_array, self.gain_parameters, data_losses_fit)
            df_help = pd.DataFrame(train_imputed_data, columns=train_help_df.columns, index=train_help_df.index)
            d[cur_dq_ml] = {i: {"train": df_help[f"{config.colname_raw}_pred"]}}

            self.model = gain

        except Exception as e:
            print(f"Exception encountered: {e}")
            d[cur_dq_ml] = {
                i: {
                    "train": df_fea[config.colname_raw],
                    "test": df_fea[config.colname_raw]
                }
            }

    def predict(self, df_fea, df_flag_erroneous, config):
        data_losses = 1

        # Test Data Imputation
        test_help_df = df_fea.fillna(-7777)
        test_array = test_help_df.to_numpy(dtype="float")
        test_imputed_data = gain(test_array, self.gain_parameters, data_losses)
        df_help = pd.DataFrame(test_imputed_data, columns=test_help_df.columns, index=test_help_df.index)
        return df_help[f"{config.colname_raw}_pred"]

    def save_model(self, path):
        """
        Save the model to a file.

        Parameters
        ----------
        path : str
          The path to save the model file.
        """
        joblib.dump(self.model, path)

    def load_model(self, path):
        """
        Load the model from a file.

        Parameters
        ----------
        path : str
          The path to the model file.
        """
        self.model = joblib.load(path)

    '''
    try:

    # GAIN+dense loss
    # before putting into own package check with licenses!!!
    # https://github.com/jsyoon0823/GAIN
    # TODO: extra fit and transform methods missing. Right now only fit_transform part
    # therefore also newly trained model for test dataset!! :(
    gain_parameters = {'batch_size': 128,  # default param from author of GAIN
                       'hint_rate': 0.9,  # default param from author of GAIN
                       'alpha': 100,  # default param from author of GAIN
                       'iterations': 10  # default param from author of GAIN
                       }
    dw = DenseWeight(alpha=1.0)
    data_losses_train = 1  # dw.fit(train_["value_plaus"].to_numpy(dtype="float"))
    data_losses_test = 1  # dw.fit(test_["value_plaus"].to_numpy(dtype="float"))

    train_help_df = pd.concat([dataSetHandler.get_train_features(detMLConfig.exclude_cols)[i].fillna(-7777), dataSetHandler.get_train_features()[i][["value_plaus_pred"]]], axis = 1)
    train_array = train_help_df.to_numpy(dtype="float")
    train_imputed_data = gain.gain(train_array, gain_parameters, data_losses_train)
    df_help = pd.DataFrame(train_imputed_data)
    df_help.columns = train_help_df.columns
    df_help.index = train_help_df.index
    d[cur_dq_ml][i]["train"] = df_help["value_plaus_pred"]

    test_help_df = pd.concat([dataSetHandler.get_test_features(detMLConfig.exclude_cols)[i].fillna(-7777), dataSetHandler.get_test_features()[i][["value_plaus_pred"]]], axis = 1)
    test_array = test_help_df.to_numpy(dtype="float")
    test_imputed_data = gain.gain(test_array, gain_parameters, data_losses_test)
    df_help = pd.DataFrame(test_imputed_data)
    df_help.columns = test_help_df.columns
    df_help.index = test_help_df.index
    d[cur_dq_ml][i]["test"] = df_help["value_plaus_pred"]

except:
    d[cur_dq_ml][i]["train"] = dataSetHandler.get_train_features(detMLConfig.exclude_cols)[i][detMLConfig.colname_raw]
    d[cur_dq_ml][i]["test"] = dataSetHandler.get_test_features(detMLConfig.exclude_cols)[i][detMLConfig.colname_raw]

    pass
    '''

class ML_byRF:
    """
    This class implements a Random Forest model for regression tasks using the RandomForestRegressor from
    scikit-learn.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest, influencing the model's complexity and performance.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator for reproducibility.
    max_depth : int, default=15
        The maximum depth of the trees. It helps prevent overfitting by limiting how deep the trees can grow.
    """
    def __init__(self, n_estimators=100, random_state=None, max_depth = 15):
        r"""
        Initializes the ML_byRF class with a RandomForestClassifier model.

        Parameters
        ----------
        n_estimators : int, default=100
            The number of trees in the forest.
        random_state : int, RandomState instance or None, default=None
            Controls the randomness of the estimator.
        """
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, max_depth=max_depth)

    def fit(self, df_fea, df_tar, df_flag_erroneous, config):
        r"""
        Fits the RandomForestClassifier model to the data. Function does not support NaN values.

        Parameters
        ----------
        df_fea : pandas dataframe
            The training input samples.
        df_tar : pandas dataframe
            The target values (class labels).

        Returns
        -------
        decision_clf : RandomForestClassifier object


        """
        train_ = df_fea.fillna(-777)
        train_part = train_[train_[df_flag_erroneous] == True].copy()
        test_ = df_tar.fillna(-777)
        test_part = test_[train_[df_flag_erroneous] == True].copy()
        self.model.fit(train_part, test_part)

    def predict(self, df_fea, df_flag_erroneous, config):#, model):
        r"""
        Predict class for X. Function does not support NaN values.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array of shape (n_samples,)
          The predicted classes.
        """
        r"""
        Function does not support NaN values.

        Parameters
        ----------
        df : pandas dataframe
        exclude_cols : list columns not to regard in model
        decision_clf : RandomForestClassifier object

        Returns
        -------
        hasGoodDQ : pandas series

        Examples
        --------
        >>> ML_byRF_fromModel(df[["col1"]], [], decision_clf)
        0     True
        1     True
        2     True
        3     True
        4     True
        5     True
        6     True
        7     True
        8     True
        9    False
        dtype: bool
        """


        train_ = df_fea.fillna(-777)
        train_part = train_[train_[df_flag_erroneous] == True].copy()
        if train_part.shape[0]==0:
            return None
        train_part[f"{config.colname_raw}_pred"] = self.model.predict(train_part)
        # retain the value from config.colname_raw if df_flag_erroneous is False, otherwise set it to np.nan
        df_fea[f"{config.colname_raw}_pred"] = np.where(~df_fea[df_flag_erroneous], df_fea[config.colname_raw], np.nan)
        return df_fea[f"{config.colname_raw}_pred"].fillna(train_part[f"{config.colname_raw}_pred"])

    def save_model(self, path):
        """
        Save the model to a file.

        Parameters
        ----------
        path : str
          The path to save the model file.
        """
        joblib.dump(self.model, path)

    def load_model(self, path):
        """
        Load the model from a file.

        Parameters
        ----------
        path : str
          The path to the model file.
        """
        self.model = joblib.load(path)


class ML_byMissForest:
    r"""
    This class implements the MissForest algorithm for imputation of missing values
    using Random Forests. It is designed to handle datasets with missing entries,
    providing a mechanism to fill in these gaps effectively.

    Parameters
    ----------
    max_features : int, default=100
        The maximum number of features to consider when looking for the best split
        at each node in the forest. This can help in controlling overfitting and
        improving computational efficiency.
    """
    def __init__(self, max_features=100):
        """
        Initializes the ML_byRF class with a RandomForestClassifier model.

        Parameters
        ----------
        - n_estimators: int, default=100
          The number of trees in the forest.
        - random_state: int, RandomState instance or None, default=None
          Controls the randomness of the estimator.
        """
        self.model = missforest.MissForest(criterion="squared_error", max_features=max_features)

    def fit(self, df_fea, df_tar, df_flag_erroneous, config):
        r"""
        Fits the MissForest model to the training data for imputation of missing values.

        This method prepares the feature DataFrame for training the MissForest model. It
        replaces missing values based on the specified criteria and fits the model to the
        non-empty feature columns.

        Parameters
        ----------
        df_fea : pandas DataFrame
            The input features with missing values that need to be imputed.
        df_tar : pandas DataFrame
            The target values (not used in MissForest but can be included for consistency).
        df_flag_erroneous : pandas Series
            A boolean Series indicating which rows contain erroneous or missing values.
        config : object
            Configuration object containing information about the dataset, including
            the name of the column to be predicted.

        Returns
        -------
        None

        """
        if config.dataset_size == "large":
            # do not run this method, takes very long to compute
            print("Model ML_byGAIN_DW cannot be computed.")
        try:
            # retain the value from config.colname_raw if isValid_pred is True, or set it to np.nan if isValid_pred is False
            df_fea[f"{config.colname_raw}_pred"] = np.where(~df_fea[df_flag_erroneous], df_fea[config.colname_raw], np.nan)
            train_ = df_fea.dropna(axis="columns", how="all")
            '''
            def intersection(lst1, lst2):
                lst3 = [value for value in lst1 if value in lst2]
                return lst3

            used_cols = intersection(train_.columns, test_.columns)
            train_ = train_[used_cols]
            test_ = test_[used_cols]
            '''

            self.model.fit_transform(train_)

        except Exception as e:
            print(e)
            print("MissForest not working properly.")

    def predict(self, df_fea, df_flag_erroneous, config):  # , model):
        r"""
        Imputes missing values in the input feature DataFrame using the fitted MissForest model.

        This method predicts values for the specified column in the DataFrame where the
        original values are missing (indicated by the `df_flag_erroneous`). It retains the
        original values where valid and fills in NaN where the values are flagged as erroneous.

        Parameters
        ----------
        df_fea : pandas DataFrame
            The input features with missing values to be imputed.
        df_flag_erroneous : String
            A String referring to a column of df_fea with a boolean Series indicating which rows contain erroneous or missing values.
        config : object
            Configuration object that contains information about the dataset, including
            the name of the column to be predicted.

        Returns
        -------
        pandas Series
            A Series containing the predicted values for the specified column, with
            NaN where the original values were not valid.
        """
        try:
            # retain the value from config.colname_raw if isValid_pred is True, or set it to np.nan if isValid_pred is False
            df_fea[f"{config.colname_raw}_pred"] = np.where(~df_fea[df_flag_erroneous], df_fea[config.colname_raw], np.nan)
            test_ = df_fea.dropna(axis="columns", how="all")
            X_test_filled = self.model.transform(test_)
            df_help = pd.DataFrame(X_test_filled)
            df_help.columns = test_.columns
            df_help.index = test_.index
            return df_help[f"{config.colname_raw}_pred"]
        except Exception as e:
            print(e)
            print("MissForest not working properly.")

    def save_model(self, path):
        """
        Save the model to a file.

        Parameters
        ----------
        path : str
          The path to save the model file.
        """
        joblib.dump(self.model, path)

    def load_model(self, path):
        """
        Load the model from a file.

        Parameters
        ----------
        path : str
          The path to the model file.
        """
        self.model = joblib.load(path)