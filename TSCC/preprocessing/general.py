import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from denseweight import DenseWeight
import resreg
import smogn
from imblearn.over_sampling import SMOTENC
import datetime


class Config:
    r"""
    colname_target_det : str, optional
        The column name of the target variable for detection models.
    colname_target_corr : str, optional
        The column name of the target variable for correction models.
    colname_raw : str, optional
        The column name of the raw data to be used as input.
    colname_id : str, optional
        The column name containing unique identifiers for each data point or entity.
    colname_isErroneousPred : str, optional
        The column name indicating whether a prediction considers an observation  erroneous.
    exclude_cols : list, optional
        A list of column names to exclude from processing.
    colname_isEvent : str, optional
        The column name indicating whether an event occurred (used in event-based modeling).
    sensortype : str, optional
        Specifies the type of sensor used in data collection.
    target_sensor_uncertainty : float, optional
        The uncertainty associated with the target sensor measurement strategy.
    frequency : str, optional
        The frequency of the time series data.
    det_ML_methods : list, optional
        A list of machine learning methods for detection.
    det_stat_methods : list, optional
        A list of statistical methods for detection.
    corr_ML_methods : list, optional
        A list of machine learning methods for correction.
    corr_stat_methods : list, optional
        A list of statistical methods for correction.
    cross_validation : bool, default False
        Whether to perform cross-validation during model training.
    train_test_IDs : list, optional
        A list of IDs used to split the dataset into training and testing sets.
    threshold_nrObs : int, default 10000
        The threshold number of observations required to perform analysis.
    detMethod_for_corr : str, optional
        The detection method used for data correction.
    random_state : int, default 1
        The seed for random number generation to ensure reproducibility.
    dataset_size : str, default "small"
        The size of the dataset, used to optimize computation resources.
    det_ML_model_savefolder : str, optional
        The folder path to save detection machine learning models.
    corr_ML_model_savefolder : str, optional
        The folder path to save correction machine learning models.
    """

    def __init__(self,
                 colname_target_det=None,
                 colname_target_corr=None,
                 colname_raw=None,
                 colname_id=None,
                 colname_isErroneousPred=None,
                 exclude_cols=None,
                 colname_isEvent=None,
                 sensortype=None,
                 target_sensor_uncertainty=None,
                 frequency = None,
                 cross_validation=False,
                 train_test_IDs=None,
                 threshold_nrObs=10000,
                 detMethod_for_corr = None,
                 random_state = 1,
                 dataset_size = "small",
                 det_ML_model_savefolder = None,
                 corr_ML_model_savefolder = None):
        self.cross_validation = cross_validation
        self.train_test_IDs = train_test_IDs
        self.colname_target_det = colname_target_det
        self.sensortype = sensortype
        self.target_sensor_uncertainty = target_sensor_uncertainty
        self.frequency = frequency
        self.threshold_nrObs = threshold_nrObs
        self.exclude_cols = exclude_cols
        self.colname_target_corr = colname_target_corr
        self.df_id_column = colname_id
        self.colname_isErroneousPred = colname_isErroneousPred
        self.colname_isEvent = colname_isEvent
        self.colname_raw = colname_raw
        self.detMethod_for_corr = detMethod_for_corr
        self.random_state = random_state
        self.dataset_size = dataset_size
        self.det_ML_model_savefolder = det_ML_model_savefolder
        self.corr_ML_model_savefolder = corr_ML_model_savefolder

    def update_frequency(self, frequency):
        self.frequency = frequency

def get_train_test_split(df, variation = "random", test_size = 0.2, cv_parameters = [5, 12], random_state = 0):
    if variation == "random":
        train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    elif variation == "time_based":
        split_index = int(df.shape[0]*(1-test_size))

        train = df.iloc[:split_index]
        test = df.iloc[split_index:]
    elif variation == "cv_time_based":
        cv_folds = cv_parameters[0]
        nrObsPerFold = df.shape[0]/cv_folds
        padding = cv_parameters[1]
        #df = df.reset_index(drop = True)
        train = []
        test = []
        for i in range(cv_folds):
            min_idx_test = int(nrObsPerFold*(i))
            max_idx_test = int(nrObsPerFold*(i+1))
            cur_test = df[min_idx_test:max_idx_test]
            test.append(cur_test)
            train.append(df.drop(index=cur_test.index))
    else:
        train = pd.DataFrame()
        test = pd.DataFrame()
    return train, test

def train_test_fea_tar_split(df_model,
                             config,
                             tar_columns):
    r"""
    Generate list of train and test datasets.
    results in one entry per list if cross_validation is False

    Parameters
    ----------
    df_model : pandas dataframe
        including all features, target, and supplementary columns
    cross_validation : boolean
        ...
    train_test_IDs : ...
        only possible when ``cross_validation`` is False

    Returns
    -------
    train_list : list
        list of train datasets
    df_list : list
        list of test datasets

    """

    # perform train test split
    if config.cross_validation:
        train_list, test_list = get_train_test_split(df_model.fillna(-777), test_size = 0.2,#.reset_index(drop=True),
                                                     variation="cv_time_based")
        #for i, cur_df in enumerate(train_list):
        #    train_list[i] = cur_df#.reset_index(drop=True)
        #for i, cur_df in enumerate(test_list):
        #    test_list[i] = cur_df#.reset_index(drop=True)
    elif isinstance(config.train_test_IDs, list):
        train = df_model[df_model[df_id_column].isin(train_test_IDs[0])].fillna(-777)
        train_list = [train]#.reset_index(drop=True)]
        df = df_model[df_model[df_id_column].isin(train_test_IDs[1])].fillna(-777)
        test_list = [df]#.reset_index(drop=True)]
    else:
        train, df = get_train_test_split(df_model.fillna(-777),#.reset_index(drop=True),
                                         variation="time_based")
        train_list = [train]#.reset_index(drop=True)]
        test_list = [df]#.reset_index(drop=True)]

    # perform fea tar exclude split
    train_list_fea = [df_train.drop(columns = tar_columns, errors = "ignore") for df_train in train_list]
    train_list_tar = [df_train[tar_columns] for df_train in train_list]
    test_list_fea = [df_train.drop(columns = tar_columns, errors = "ignore") for df_train in test_list]
    test_list_tar = [df_train[tar_columns] for df_train in test_list]#[detMLConfig.df_id_column] +


    return train_list_fea, train_list_tar, test_list_fea, test_list_tar


def undersampling_valrange(df_fea, df_tar, colname, undersampling_rate, val_range, inclusive_range = "both"):
    r"""

    Imbalanced data method

    Parameters
    ----------
    df : pandas data frame
        Data frame to be undersampled
    undersampling_rate : float
        Ranges in [0, 1]
    colname : string
        Column name to be undersampled
    val_range : list
        List of min, max value range
    inclusive_range : string
        inclusive_range "both" means [val_range[0], val_range[1]]

    Returns
    -------
    u_df_fea: pandas data frame
        The undersampled target dataframe.
    u_df_tar: pandas data frame
        The undersampled target dataframe.

    """
    df = pd.concat([df_fea, df_tar], axis = 1)
    df_help = df[df[colname].between(val_range[0], val_range[1], inclusive = inclusive_range)]
    undersampling_nr = np.ceil(df_help.shape[0]*undersampling_rate)#(df.shape[0] - df_help.shape[0])
    df_help = df_help.sample(n = int(undersampling_nr))
    df = pd.concat([df[~df[colname].between(val_range[0], val_range[1], inclusive = inclusive_range)], df_help]).\
        sort_index()

    u_df_fea = df_fea.join(df[[]], how = "right")
    u_df_tar = df_tar.join(df[[]], how = "right")
    return u_df_fea, u_df_tar

def timstamp_remove_duplicates(df, colname_id):
    # Find duplicates in the 'timestamp' column
    df = df.sort_values(by = colname_id)
    dups = df.duplicated(subset=[colname_id], keep=False)

    # Initialize a counter for each group of duplicates
    dup_counter = {}

    # Function to add seconds to duplicates
    def add_seconds_to_duplicates(row):
        ts = row[colname_id]

        if ts in dup_counter:
            dup_counter[ts] += 1
        else:
            dup_counter[ts] = 0

        return ts + pd.Timedelta(milliseconds=dup_counter[ts])#datetime.timedelta


    # Apply the function to duplicates
    df.loc[dups, colname_id] = df[dups].apply(add_seconds_to_duplicates, axis=1)

    return df

def SMOTE(df_fea, df_tar, colname_target, colname_id, exclude_columns, random_state=42):
    """
    Apply the SMOTE (Synthetic Minority Over-sampling Technique) to handle imbalanced datasets.

    Parameters
    ----------
    df_fea : pandas DataFrame
        The input feature dataframe containing independent variables.
    df_tar : pandas DataFrame
        The target dataframe containing the dependent variable(s).
    colname_target : str
        The column name of the target variable that requires oversampling.
    colname_id : str
        The column name used as the unique identifier for each instance in the data.
    exclude_columns : list
        A list of column names to be excluded from the SMOTE process.
    random_state : int, optional
        The seed used by SMOTE for random number generation, default is 42.

    Returns
    -------
    smote_fea : pandas DataFrame
        The oversampled feature dataframe with the same structure as the original `df_fea`, indexed by `colname_id`.
    smote_tar : pandas DataFrame
        The oversampled target dataframe with the same structure as the original `df_tar`, indexed by `colname_id`.

    """

    from imblearn.over_sampling import SMOTE
    smote_preproc = SMOTE(random_state=random_state)
    smote_fea, smote_tar = smote_preproc.fit_resample(df_fea.drop(columns=exclude_columns, errors = "ignore").\
                                                      reset_index(drop=True).reset_index(),
                                                      df_tar[[colname_target]])

    smote_tar = df_tar.reset_index().join(smote_fea.set_index("index")[[]])
    smote_tar =  timstamp_remove_duplicates(smote_tar, colname_id). \
        set_index(colname_id)

    smote_fea = df_fea.reset_index().join(smote_fea.set_index("index")[[]])
    smote_fea = timstamp_remove_duplicates(smote_fea, colname_id). \
        set_index(colname_id)

    return smote_fea, smote_tar

def SMOTEwithCat(df_fea, df_tar, colname, random_state = 42):
    """
    Apply the SMOTE-NC (Synthetic Minority Over-sampling Technique for Nominal and Continuous features)
    to handle imbalanced datasets that include both categorical and continuous features.

    Parameters
    ----------
    df_fea : pandas DataFrame
        The input feature dataframe containing both categorical and continuous independent variables.
    df_tar : pandas DataFrame
        The target dataframe containing the dependent variable(s).
    colname : str
        The column name of the target variable that requires oversampling.
    random_state : int, optional
        The seed used by SMOTE-NC for random number generation, default is 42.

    Returns
    -------
    smote_fea : pandas DataFrame
        The oversampled feature dataframe with categorical and continuous features.
    smote_tar : pandas DataFrame
        The oversampled target dataframe.

    """

    smote_nc = SMOTENC(categorical_features = "auto", random_state=random_state)
    smote_fea, smote_tar = smote_nc.fit_resample(df_fea,
                                                   df_tar[[colname]])

    return smote_fea, smote_tar

def SMOGN(df_fea, df_tar, colname, colname_idx, rel_thres = 0.8):
    """
    Apply the SMOGN (Synthetic Minority Over-sampling Technique for Regression with Gaussian Noise)
    to handle imbalanced regression datasets by oversampling both rare and extreme values.

    Parameters
    ----------
    df_fea : pandas DataFrame
        The input feature dataframe containing the independent variables.
    df_tar : pandas DataFrame
        The target dataframe containing the dependent variable(s).
    colname : str
        The column name of the target variable that requires oversampling.
    colname_idx : str
        The column name that serves as the unique identifier for each instance in the data.
    rel_thres : float, optional
        The relevance threshold for identifying rare and extreme values, default is 0.8.

    Returns
    -------
    smogn_fea : pandas DataFrame
        The oversampled feature dataframe with rare and extreme values oversampled, indexed by `colname_idx`.
    smogn_tar : pandas DataFrame
        The oversampled target dataframe with rare and extreme values oversampled, indexed by `colname_idx`.

    """

    cur_df = pd.concat([df_fea, df_tar])
    # smogn fails seemingly-randomly
    n_tries = 0
    done = False
    smogn_fea = pd.DataFrame()
    smogn_tar = pd.DataFrame()
    while not done:
        try:
            train_part_smogn = smogn.smoter(
                data=cur_df.dropna(subset = [colname]).reset_index(),
                y=colname,
                rel_thres=rel_thres,
                # rel_coef = 2.25
            )
            smogn_fea = df_fea.join(train_part_smogn[[colname_idx]].set_index(colname_idx), how="right")
            smogn_tar = df_tar.join(train_part_smogn[[colname_idx]].set_index(colname_idx), how="right")

            done = True
        except Exception as e:
            print(e)
            print(f"rf SMOGN try {n_tries} cannot be executed")
            done = True

    return smogn_fea, smogn_tar

def check_equidistant_minute_timestamps(df):
    r"""
    Check if the datetime index of a DataFrame has equidistant time stamps each minute.

    Parameters:
    df (pd.DataFrame): DataFrame with a datetime index.

    Returns:
    bool: True if time stamps are equidistant each minute, False otherwise.
    """

    '''
    Example:
    # Sample data
    data = {'value': [1, 2, 3, 4, 5]}
    index = pd.to_datetime(['2024-07-23 10:00', '2024-07-23 10:01', '2024-07-23 10:02', '2024-07-23 10:03', '2024-07-23 10:04'])
    df = pd.DataFrame(data, index=index)
    '''
    # Calculate the difference between consecutive time stamps
    time_diffs = df.index.to_series().diff()

    # Check if all differences are equal to one minute
    is_equidistant = (time_diffs[1:] == pd.Timedelta(minutes=1)).all()

    return is_equidistant

def check_equidistant_problems(df, frequency='T'):
    """
    Check problems in a DataFrame regarding equidistant datetime index values.

    Parameters:
    df (pd.DataFrame): DataFrame with a datetime index.
    frequency (str): Frequency string (e.g., 'T' for minute, 'H' for hour).

    Returns:
    dict: Dictionary with the problems found in the dataset.
    """

    ''' 
    # Sample data
    data = {'value': [1, 2, 3, 4, 5]}
    index = pd.to_datetime(['2024-07-23 10:00', '2024-07-23 10:01', '2024-07-23 10:02', '2024-07-23 10:04', '2024-07-23 10:04'])
    df = pd.DataFrame(data, index=index)
    
    # Check for problems
    problems = check_equidistant_problems(dataSetHandler.df)
    
    # Print the result
    print("Missing timestamps:", problems['missing_timestamps'])
    print("Duplicate timestamps:", problems['duplicate_timestamps'])
    print("Non-equidistant timestamps:", problems['non_equidistant_timestamps'])
    
    duplicates not entirely correct, timestamp only at beinning of day, prob. wrong dataSetHandler.df[isDup].index.strftime('%Y-%m-%d %H:%M:%S')
    '''
    problems = {
        'missing_timestamps': [],
        'duplicate_timestamps': [],
        'non_equidistant_timestamps': False
    }

    # Ensure the index is sorted
    df = df.sort_index()

    # Full range of timestamps
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=frequency)

    # Find missing timestamps
    missing_timestamps = full_range.difference(df.index)
    if not missing_timestamps.empty:
        problems['missing_timestamps'] = missing_timestamps

    # Find duplicate timestamps
    duplicate_timestamps = df.index[df.index.duplicated(keep="first")].strftime('%Y-%m-%d %H:%M:%S')
    if not duplicate_timestamps.empty:
        problems['duplicate_timestamps'] = duplicate_timestamps

    # Check for non-equidistant timestamps
    time_diffs = df.index.to_series().diff().dropna()
    expected_diff = pd.to_timedelta(f'1{frequency}')
    if not (time_diffs == expected_diff).all():
        problems['non_equidistant_timestamps'] = True

    return problems


class DataSetHandler:
    r"""

    DataSetHandler class for handling data split into train and test as
    well as resampling operations.

    Relies on
    ----------
    cross_validation : boolean
        ...
    train_test_IDs : ...
        only possible when ``cross_validation`` is False
    exclude_cols : list columns not to regard in model

    Parameters
    ----------
        - data (pd.DataFrame): The input DataFrame.
        - test_size (float): The proportion of the dataset to include in the test split.
        - random_state (int): Random seed for reproducibility.

    Attributes
    ----------
    df_train_list_fea : list of pandas DataFrame
        List of training feature DataFrames.
    df_train_list_tar : list of pandas DataFrame
        List of training target DataFrames.
    df_train_list_excl : list of pandas DataFrame
        List of excluded columns in training DataFrames.
    df_test_list_fea : list of pandas DataFrame
        List of testing feature DataFrames.
    df_test_list_tar : list of pandas DataFrame
        List of testing target DataFrames.
    df_test_list_excl : list of pandas DataFrame
        List of excluded columns in testing DataFrames.
    list_len : int
        Length of the training feature DataFrame list.
    """

    def __init__(self, df, config):
        if check_equidistant_minute_timestamps(df):
            self.df = df
        else:
            if not config.frequency:
                frequency = pd.Series(df.index).diff().mode()[0]
                config.update_frequency(frequency)
            else:
                frequency = config.frequency
            problems = check_equidistant_problems(df, frequency=frequency)
            #full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=detMLConfig.frequency)
            #df = pd.DataFrame([np.nan] * len(full_range), index=full_range, columns=["placeholder"]).join(df).drop(
            #    columns=["placeholder"])
            #print("Added missing timestamps:", problems['missing_timestamps'])
            isDup = df.index.duplicated(keep="first")
            self.df = df[~isDup]
            print("Removed duplicate timestamps:", problems['duplicate_timestamps'])

        # self.test_size = detMLConfig.test_size
        self.config = config
        self.random_state = config.random_state
        self.tar_columns = [config.colname_target_det, config.colname_target_corr]
        # self._split_data()

        self.train_list_fea, self.train_list_tar, \
            self.test_list_fea, self.test_list_tar = train_test_fea_tar_split(self.df, config,
                                                                              self.tar_columns)

        self.df_results = self.df[[config.colname_raw]]

        self.list_len = len(self.train_list_fea)
        self.train_event_absnr = [cur_df[config.colname_isEvent].sum() if config.colname_isEvent else np.nan for cur_df in self.train_list_fea ]
        self.train_eventrates = [(self.train_event_absnr[i]/cur_df.shape[0]) for i, cur_df in enumerate(self.train_list_fea)]

    def getCharacteristics(self, detMLConfig):
        d = {}
        d["train_eventrate"] = [df[detMLConfig.colname_isEvent].mean() for df in df_train_list_excl]
        d["train_event_absnr"] = [df[detMLConfig.colname_isEvent].sum() for df in df_train_list_excl]
        d["Detection_method"] = np.nan
        return d

    def preprocess_training_ds(self, preproc_fct, *args):

        cur_train_list_fea = self.train_list_fea
        #if exclude_columns:
        #    cur_train_list_fea = [df.drop(columns=exclude_columns, errors = "ignore") for df in self.train_list_fea]

        preproc_fea_list = []
        preproc_tar_list = []
        for i in range(len(cur_train_list_fea)):
            preproc_fea, preproc_tar = preproc_fct(cur_train_list_fea[i],
                                                   self.train_list_tar[i],
                                                   *args)
            preproc_fea_list.append(preproc_fea)
            preproc_tar_list.append(preproc_tar)

        self.train_list_fea = preproc_fea_list
        self.train_list_tar = preproc_tar_list

    def get_train_features(self, exclude_columns=None):  # get_df_train_list_fea
        """
        Returns the list of training DataFrames for features.

        Returns
        -------
        list of pandas DataFrame
        """

        cur_train_list_fea = self.train_list_fea

        if exclude_columns:
            cur_train_list_fea = [df.drop(columns=exclude_columns, errors = "ignore") for df in self.train_list_fea]

        return cur_train_list_fea

    def get_train_targets(self, extract_single_target=None):
        """
        Returns the list of training DataFrames for targets.

        Returns
        -------
        list of pandas DataFrame
        """
        if extract_single_target:
            return [df[extract_single_target] for df in self.train_list_tar]
        return self.train_list_tar

    def get_test_features(self, exclude_columns=None):
        """
        Returns the list of testing DataFrames for features.

        Returns
        -------
        list of pandas DataFrame
        """
        if exclude_columns:
            return [df.drop(columns=exclude_columns, errors = "ignore") for df in self.test_list_fea]
        return self.test_list_fea

    def get_test_targets(self, extract_single_target=None):
        """
        Returns the list of testing DataFrames for targets.

        Returns
        -------
        list of pandas DataFrame
        """
        if extract_single_target:
            return [df[extract_single_target] for df in self.test_list_tar]
        return self.test_list_tar

    def get_complete_dataset(self, exclude_columns=None):
        """
        Return the original data, optionally excluding specific columns.

        Parameters :
        exclude_columns (list) : List of columns to exclude from the original data.
        """
        if exclude_columns:
            return self.df.drop(columns=exclude_columns, errors = "ignore")
        return self.df

    def append_col_to_df_train_test(self, series):
        r"""
        Returns the list of training DataFrames for features with an additional column.

        Parameters
        ----------
        series : pandas.Series
            The series to be appended as a new column to the DataFrame.

        Returns
        -------
        list of pandas DataFrame
            A list of DataFrames for training with the additional column appended.
        """

        self.df = self.df.join(series)

        for i in range(self.list_len):
            self.train_list_fea[i] = self.train_list_fea[i].join(series, how='left', validate = "one_to_one")
            self.test_list_fea[i] = self.test_list_fea[i].join(series, how='left', validate = "one_to_one")

    def append_col_to_df_results(self, series):
        """
        Append a new column to the DataFrame.
        The new column should have a name.
        """
        self.df_results = self.df_results.join(series)#data_observations



