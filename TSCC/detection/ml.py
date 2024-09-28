# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

"""
The following functions use a simple dataframe, so you can understand what
the variables may look like in your own project. But we create immediately two dataframes,
because some functions need an extra time index.

First dataframe
---------------

>>> d = {'col1': [1, 9, 3, 6, 20, 8, 3, 1, 30, 5]}
>>> df = pd.DataFrame(data=d)
	col1
0	1
1	9
2	3
3	6
4	20
5	8
6	3
7	1
8	30
9	5

Second dataframe
----------------

>>> dates = np.arange('2021-07-24 00:00:00', '2021-07-24 10:00:00', dtype='datetime64[h]')
>>> d = {'col1': [1, 9, 3, 6, 20, 8, 3, 1, 30, 5]}
>>> df_time = pd.DataFrame(data=d, index=dates)
	                col1
2021-07-24 00:00:00	1
2021-07-24 01:00:00	9
2021-07-24 02:00:00	3
2021-07-24 03:00:00	6
2021-07-24 04:00:00	20
2021-07-24 05:00:00	8
2021-07-24 06:00:00	3
2021-07-24 07:00:00	1
2021-07-24 08:00:00	30
2021-07-24 09:00:00	5
"""

class ML_byOneClassSVM:
    r"""
    A class that encapsulates the functionality of a OneClassSVM model for anomaly detection and provides methods for
    reducing training samples, fitting the model, and making predictions.

    The class is designed to handle large datasets and includes functionality for reducing the number of training
    samples by undersampling based on conditions such as sensor type and the distribution of target values.
    It uses a OneClassSVM model to detect anomalies in time series data.

    Parameters
    ----------
    threshold_nrObs : int, default=10000
        The maximum number of observations to be used for training. If the dataset exceeds this threshold, undersampling or random sampling is applied.

    Attributes
    ----------
    model : OneClassSVM
        The OneClassSVM model used for anomaly detection.

    Methods
    -------
    reduce_training_samples(df_fea, gt_series_num, config)
        Reduces the number of training samples based on ground truth values, sensor type, and predefined thresholds.
    fit(df_fea, df_tar, config)
        Fits the OneClassSVM model to the input features.
    predict(df_fea)
        Predicts whether each observation in the input dataframe is an anomaly.
    save_model(path)
        Saves the trained OneClassSVM model to a specified path.
    load_model(path)
        Loads a previously saved OneClassSVM model from a file.
    """

    def __init__(self, threshold_nrObs=10000):
        r"""
        Initializes the ML_byOneClassSVM class with a default threshold for the number of observations.
        """
        self.threshold_nrObs = threshold_nrObs
        self.model = OneClassSVM(gamma='auto')

    def reduce_training_samples(self, df_fea, gt_series_num, config):
        r"""
        Reduces the number of training samples in the dataframe based on specific conditions and performs undersampling.

        This function performs undersampling on the input dataframe based on the target values (`gt_series_num`) and sensor type
        specified in the `config`. The function supports precipitation-type data by undersampling within specified ranges of
        ground truth values and target detection. The number of samples is reduced to match a predefined threshold.

        If the size of the filtered dataframe still exceeds the threshold, the function reduces it further by random sampling
        or concatenating a small subset of the original dataframe.

        Parameters
        ----------
        df_fea : pandas DataFrame
            The input features for training.
        gt_series_num : pandas Series or ndarray
            The target ground truth values (used for filtering).
        config : object
            Configuration object that includes sensor type and column names.

        Returns
        -------
        df_filtered : pandas DataFrame
            The dataframe with reduced samples, after applying undersampling based on conditions.
        """
        df_filtered = df_fea.copy()
        df_filtered["gt_series_num"] = gt_series_num

        target_corr_isNumerical = np.issubdtype(df_filtered["gt_series_num"].dtype, np.number)

        if target_corr_isNumerical & (df_filtered.shape[0] > self.threshold_nrObs) & (
                config.sensortype == "precipitation"):
            df_filtered = self.undersampling_valrange(df_filtered, 0.001, "gt_series_num", [-1, 0],
                                                      inclusive_range="right")
            df_filtered = self.undersampling_valrange(df_filtered, 0.1, "gt_series_num", [0, 1.25],
                                                      inclusive_range="right")
            df_filtered = self.undersampling_valrange(df_filtered, 0.1, "gt_series_num", [-777, -1],
                                                      inclusive_range="both")

        if df_filtered.shape[0] > self.threshold_nrObs:
            df_filtered[config.colname_target_det] = df_filtered[config.colname_target_det].astype(int)
            df_filtered = self.undersampling_valrange(df_filtered, 0.5, config.colname_target_det, [0.99, 1],
                                                      inclusive_range="right")
            df_filtered[config.colname_target_det] = df_filtered[config.colname_target_det].astype(bool)

        if df_filtered.shape[0] > self.threshold_nrObs:
            df_filtered = df_filtered.sample(n=self.threshold_nrObs)

        if int(self.threshold_nrObs / 100) < df_filtered.shape[0]:
            df_filtered = pd.concat([df_filtered, df_fea.sample(n=int(self.threshold_nrObs / 100))])
        df_filtered = df_filtered.drop(columns=["gt_series_num"])

        return df_filtered

    def fit(self, df_fea, df_tar = None, config = None):
        r"""
        Fits the OneClassSVM model to the provided dataframe.

        Parameters
        ----------
        df_fea : pandas dataframe

        """

        self.model.fit(df_fea)

    def predict(self, df_fea):
        r"""
        Predict values based on the OneClassSVM model to the provided dataframe.
        Function does not support NaN values.

        Parameters
        ----------
        df : pandas dataframe
        oneClass_SVM : OneClassSVM object

        Returns
        -------
        hasGoodDQ : pandas series
        """

        noOutlier = pd.Series(self.model.predict(df_fea), index=df_fea.index)
        #hasGoodDQ = [False if cur_val == -1 else True for cur_val in noOutlier]
        isError = [True if cur_val == -1 else False for cur_val in noOutlier]

        return pd.Series(isError, index=df_fea.index, name = "isError").astype(float)

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


class ML_byIsolationForest:
    r"""
    A class that encapsulates the functionality of an Isolation Forest model for detecting anomalies and provides
    methods for fitting the model, predicting outliers, and saving/loading the trained model.

    The class is designed for unsupervised anomaly detection in time series or tabular data and leverages the
    Isolation Forest algorithm, which isolates anomalies by partitioning data points in a random forest structure.
    It supports configurations for model parameters like the number of estimators and contamination rate.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of base estimators (trees) in the Isolation Forest ensemble.
    max_samples : int, float, or 'auto', default='auto'
        The number of samples to draw from the input data to train each base estimator.
    contamination : 'auto' or float, default='auto'
        The proportion of outliers in the dataset. If 'auto', the contamination is inferred.
    random_state : int, RandomState instance, or None, default=None
        Controls the randomness of the algorithm for reproducibility.

    Attributes
    ----------
    model : IsolationForest
        The underlying Isolation Forest model used for anomaly detection.

    Methods
    -------
    fit(df_fea, df_tar, config)
        Fits the Isolation Forest model to the provided feature data.
    predict(df_fea)
        Predicts whether each observation in the input dataframe is an anomaly.
    save_model(path)
        Saves the trained Isolation Forest model to a specified file path.
    load_model(path)
        Loads a previously saved Isolation Forest model from a file.
    """
    def __init__(self, n_estimators=100, max_samples='auto', contamination='auto', random_state=None):
        r"""
        Initialize the IsolationForest model.
        """

        self.model = IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                                     contamination=contamination, random_state=random_state)

    def fit(self, df_fea, df_tar = None, config = None):
        r"""
        Fits the Isolation Forest model to the input feature data.

        This method trains the Isolation Forest model on the provided dataset, identifying patterns in the normal data points and learning how to detect anomalies. It requires that the input data contains no NaN values, as the model cannot handle missing data.

        Parameters
        ----------
        df_fea : pandas DataFrame or pandas Series
            The feature data on which to train the model. This data should not contain any missing values (NaN).
        df_tar : pandas DataFrame or pandas Series, optional
            The target data (if any) is not used in unsupervised anomaly detection but can be passed for consistency in the API. Default is None.
        config : object, optional
            Configuration object (if any) is not used in unsupervised anomaly detection but can be passed for consistency in the API. Default is None.

        Returns
        -------
        None
            This method fits the Isolation Forest model and updates the model's internal state.

       """
        ser_df = df_fea

        if isinstance(ser_df, pd.DataFrame):
            my_array = np.array(ser_df)
        elif isinstance(ser_df, pd.Series):
            my_array = np.array(ser_df).reshape(-1, 1)

        self.model.fit(my_array)

    def predict(self, df_fea):
        r"""
        Predict values based on the Isolation Forest model to the provided dataframe.
        Function does not support NaN values.

        Parameters
        ----------
        df_fea : pandas dataframe or pandas series

        Returns
        -------
        hasGoodDQ : pandas series
        """

        ser_df = df_fea

        if isinstance(ser_df, pd.DataFrame):
            my_array = np.array(ser_df)
        elif isinstance(ser_df, pd.Series):
            my_array = np.array(ser_df).reshape(-1, 1)

        noOutlier = pd.Series(self.model.predict(my_array), index=ser_df.index)
        #hasGoodDQ = [False if cur_val == -1 else True for cur_val in noOutlier]
        isError = [True if cur_val == -1 else False for cur_val in noOutlier]

        return pd.Series(isError, index=ser_df.index, name = "isError").astype(float)

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

class ML_byMLP:
    r"""
    A class for building and using a Multi-Layer Perceptron (MLP) classifier for supervised machine learning tasks.

    This class utilizes the `MLPClassifier` from the scikit-learn library to model data using a feedforward neural network.
    The architecture of the network, including the number of hidden layers and the activation function, can be customized through the constructor parameters.

    Parameters
    ----------
    hidden_layer_sizes : tuple, optional, default=(100,)
        Defines the number of neurons in each hidden layer. The length of the tuple indicates the number of hidden layers.
    activation : {'identity', 'logistic', 'tanh', 'relu'}, optional, default='relu'
        The activation function for the hidden layer.
    solver : {'lbfgs', 'sgd', 'adam'}, optional, default='adam'
        The algorithm used for weight optimization.
    alpha : float, optional, default=0.0001
        L2 regularization term to prevent overfitting.
    batch_size : int or 'auto', optional, default='auto'
        The size of minibatches for stochastic optimizers.
    learning_rate : {'constant', 'invscaling', 'adaptive'}, optional, default='constant'
        The learning rate schedule for weight updates.
    learning_rate_init : float, optional, default=0.001
        The initial learning rate.
    max_iter : int, optional, default=200
        The maximum number of iterations for training.
    random_state : int, RandomState instance or None, optional, default=None
        Controls the random seed for reproducibility.

    Methods
    -------
    fit(df_fea, df_tar, config):
        Trains the MLP model on the provided feature data and target labels.
    predict(df_fea):
        Predicts class labels for new input samples based on the trained model.
    save_model(path):
        Saves the trained model to a specified file path.
    load_model(path):
        Loads a trained model from a specified file path.
    """

    def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='adam',
                 alpha=0.0001, batch_size='auto', learning_rate='constant',
                 learning_rate_init=0.001, max_iter=200, random_state=None):
        r"""
        Initialize the MLPClassifier model.
        """
        self.model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation,
                                   solver=solver, alpha=alpha, batch_size=batch_size,
                                   learning_rate=learning_rate, learning_rate_init=learning_rate_init,
                                   max_iter=max_iter, random_state=random_state)

    def fit(self, df_fea, df_tar, config = None):
        r"""
        Trains the Multi-Layer Perceptron (MLP) classifier using the provided feature data and target labels.

        Parameters
        ----------
        df_fea : pandas DataFrame
            The input feature data used for training the model, where each row represents a sample and each column represents a feature.
        df_tar : pandas Series or array-like
            The target labels corresponding to the input feature data. Each value indicates the class label for the respective sample.
        config : object, optional
            Configuration object for any additional settings (not currently used in this method).

        """

        train_array = df_fea.to_numpy()

        self.model.fit(train_array, df_tar.to_numpy())#squeeze


    def predict(self, df_fea):
        r"""
        Predicts class labels for the given input feature data using the trained MLP classifier.

        This method uses the fitted MLP model in a class instance to generate predictions.

        Parameters
        ----------
        df_fea : pandas DataFrame
            The input feature data for which predictions are to be made. Each row should represent a sample, and each column should represent a feature.

        Returns
        -------
        pandas Series
            A Series containing the predicted class labels for each input sample. The values are cast to float for compatibility with other numerical operations.
        """
        my_array = df_fea.to_numpy()

        return pd.Series(self.model.predict(my_array), index=df_fea.index).astype(float)

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


class ML_byRF:
    r"""
   A class for implementing a Random Forest Classifier for classification tasks.

   This class utilizes the `RandomForestClassifier` from the scikit-learn library to model the relationship
   between input features and target class labels. It provides methods for training the model, making predictions,
   and saving/loading the model.

   Parameters
   ----------
   n_estimators: int, optional, default=100
       The number of trees in the forest. A higher number can improve performance but increases computation time.

   random_state: int, RandomState instance or None, optional, default=None
       Controls the randomness of the estimator for reproducibility.

   Methods
   -------
   fit(df_fea, df_tar, config):
       Fits the RandomForestClassifier model to the training data.

   predict(df_fea):
       Predicts class labels for the given input feature data.

   save_model(path):
       Saves the trained model to a specified file path.

   load_model(path):
       Loads a previously saved model from a specified file path.
   """
    def __init__(self, n_estimators=100, random_state=None):
        """
        Initializes the ML_byRF class with a RandomForestClassifier model.
        """
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def fit(self, df_fea, df_tar, config = None):
        r"""
        Trains the RandomForestClassifier model using the provided feature and target data.

        This method fits the Random Forest model to the training dataset, enabling it to learn the relationship
        between the input features and their corresponding target class labels. Note that the function does not
        handle NaN values, so the input data must be preprocessed to ensure it is free of missing values.

        Parameters
        ----------
        df_fea: pandas DataFrame
            A DataFrame containing the input features for training. Each column represents a feature, and each row
            represents a training sample.

        df_tar: pandas DataFrame or Series
            The target values (class labels) corresponding to the input features. This should be a one-dimensional
            array-like structure where each entry matches the class label for the respective training sample in `df_fea`.

        config: optional
            A configuration object that can be used to specify additional parameters for model fitting, although it
            is not utilized in this method.

        Returns
        -------
        None
            The fitted model is stored within the instance, allowing for subsequent predictions.

        """
        self.model.fit(df_fea, df_tar)

    def predict(self, df_fea):
        """
        Predicts the class labels for the given input samples using the trained RandomForestClassifier model.

        This method takes a DataFrame of input features and uses the fitted model to generate predictions for each
        sample. The predictions indicate whether each sample is classified as an outlier or not. The method does not
        support NaN values, so ensure that the input data is clean before calling this function.

        Parameters
        ----------
        df_fea: pandas DataFrame
            A DataFrame containing the input features for which predictions are to be made. Each column represents a
            feature, and each row corresponds to a sample.

        Returns
        -------
        pandas Series
            A Series containing the predicted class labels for each input sample, with the same index as
            the input DataFrame. The values are converted to floats for consistency.

        """
        isError = pd.Series(self.model.predict(df_fea), index=df_fea.index)

        return isError.astype(float)

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


