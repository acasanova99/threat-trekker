"""
    ml_encoder.py

    This module contains generic function to encode data. It is possible to invoke its methods for any of the datasets
    within this project.

    Author: Angel Casanova
    2023
"""
import logging
import math
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def label_encoder(df: DataFrame, column_name: str) -> (DataFrame, LabelEncoder):
    """ This function makes a categorical encoding for parsing string labels into numbers.
    Args:
        df (pandas.DataFrame): data frame that is going to be encoded.
        column_name (str): name of the column that is going to be encoded.
    Returns:
        df, enc (tuple): A tuple containing the dataframe and the label encoder for recovering the original class names.
    """
    enc = preprocessing.LabelEncoder()
    df[column_name] = enc.fit_transform(df[column_name])
    return df, enc


def label_twofold_encoder(df: DataFrame, column_name: str):
    """ This function makes a binary encoding for parsing string labels into numbers.
    Args:
        df (pandas.DataFrame): data frame that is going to be encoded.
        column_name (str): name of the column that is going to be encoded.
    Returns:
        df (pandas.DataFrame): The provided dataframe but with a twofold representation of the provided column
    """
    # Calculate the majority class
    majority_class = df[column_name].mode()[0]

    # Convert values to binary representation
    df[column_name] = np.where(df[column_name] == majority_class, 0, 1)
    return df


def one_hot_for_categorical(df: DataFrame, column_name: str) -> DataFrame:
    """ This function makes a one hot encoding for the unique values of a single feature.
    Args:
        df (pandas.DataFrame): data frame that is going to be encoded.
        column_name (str): name of the column that is going to be encoded.
    Returns:
        _ (array): The resulting array with the classes in one hot encoding notation.
    """
    enc = OneHotEncoder(
        categories='auto',
        drop=None,
        sparse_output=False,
        dtype=np.float64,
        handle_unknown="error"
    )
    # print('what i pass to fit-transform')
    # print(df[[column_name]])
    transformed: np.ndarray = enc.fit_transform(df[[column_name]])
    # print(transformed, transformed[0])
    df[column_name] = transformed
    # print(df.head(10))
    return df


def yorn(df: DataFrame, column_name: str) -> DataFrame:
    """ Encode a Yes or Not field. This function just parses a Python boolean as an integer.
    Args:
        df (pandas.DataFrame): data frame that is going to be encoded.
        column_name (str): name of the column that is going to be encoded.
    Returns:
        df (pandas.DataFrame): The resulting DataFrame with the encoded feature.
    """
    df[column_name] = (df[column_name]).astype(int)
    return df


def ordinal_value_of(df: DataFrame, column_name: str) -> DataFrame:
    """ Parse string categorical values with its index value within the array of columns.
    Args:
        df (pandas.DataFrame): data frame that is going to be encoded.
        column_name (str): name of the column that is going to be encoded.
    Returns:
        df (pandas.DataFrame): The resulting DataFrame with the encoded feature.
    """
    oe = OrdinalEncoder(categories=[df[column_name].unique()])
    df[column_name] = oe.fit_transform(df.loc[:, [column_name]])
    return df


def ipv4_to_int32(df: DataFrame, column_name: str) -> DataFrame:
    """ Parse an Ipv4 Address as an integer by splitting the address by its class domains and
     joining again its binary values. As indicates the following ascii plot.
        127        .        0        .        0        .        1
         |                  |                 |                 |
         v                  v                 V                 V
       8-bit               8-bit           8-bit              8-bit
         |                   |               |                 |
         +-------------------+-------+-------+-----------------+
                                     | (concat binary values)
                                     v
                                   int32
    Args:
        df (pandas.DataFrame): data frame that is going to be encoded.
        column_name (str): name of the column that is going to be encoded.
    Returns:
        df (pandas.DataFrame): The resulting DataFrame with the encoded feature.
    """

    df[column_name] = df[column_name].apply(__ipv4_to_int32)
    return df


def __ipv4_to_int32(ipv4: str) -> int:
    if ':' in ipv4:  # Avoid ipv6 for this first version
        return 0
    split_addr: list[str] = ipv4.split('.')
    return (int(split_addr[0]) << 24) + (int(split_addr[1]) << 16) + \
        (int(split_addr[2]) << 8) + int(split_addr[3])


def remove_column(df: DataFrame, column_name: str) -> DataFrame:
    """ Remove a column from the dataframe.
    Args:
        df (pandas.DataFrame): data frame that is going to be encoded.
        column_name (str): name of the column that is going to be encoded.
    Returns:
        df (pandas.DataFrame): The resulting DataFrame with the encoded feature.
    """
    df.drop(column_name, axis=1, inplace=True)
    return df


def ts_to_epoch(df: DataFrame, column_name: str) -> DataFrame:
    """ Parse a complex datetime object to an integer that stores the epoch time.
    Args:
        df (pandas.DataFrame): data frame that is going to be encoded.
        column_name (str): name of the column that is going to be encoded.
    Returns:
        df (pandas.DataFrame): The resulting DataFrame with the encoded feature.
    """
    date = datetime.utcnow()
    date64 = np.datetime64(date)
    df[column_name] = df[column_name].apply(lambda ts: (date64 - ts) / np.timedelta64(1, 's'))
    return df


def clean_dataset(df) -> DataFrame:
    """ Remove non-number, infinite, and - infinite values from the dataset.
    Args:
        df (pandas.DataFrame): data frame that is going to be filtered.
    Returns:
        df (pandas.DataFrame): The resulting DataFrame without the empty values.
    """
    logger = logging.getLogger('ThreatTrekker')
    assert isinstance(df, DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)

    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    # # noinspection PyTypeChecker
    df = df[indices_to_keep].astype(np.float64)
    logger.debug(f'There is any null value in the dataset?: {np.any(np.isnan(df))}')
    logger.debug(f'All rows in the dataset have finite values?: {np.all(np.isfinite(df))}')
    return df


def describe(df: pd.DataFrame, label: str = 'label_tactic') -> None:
    logger = logging.getLogger('ThreatTrekker')
    logger.info(f'Dataframe information about the output classes: ')
    logger.info(f'Number of elements: {df[label].nunique()}')
    logger.info(f'Element Names: {df[label].unique()}')
    logger.info(f'Number of samples per element:\n{df[label].value_counts()}')


def balance(df: pd.DataFrame, target_variable: str, undersample_dict: dict, oversample_dict: dict) -> pd.DataFrame:
    logger = logging.getLogger('ThreatTrekker')
    # split your dataframe into features and target variables
    X = df.drop(target_variable, axis=1)
    y = df[target_variable]

    # apply under-sampling to majority classes
    undersample = RandomUnderSampler(sampling_strategy=undersample_dict, random_state=42)
    X_undersampled, y_undersampled = undersample.fit_resample(X, y)
    aux = pd.concat(
        [pd.DataFrame(X_undersampled), pd.DataFrame(y_undersampled, columns=[target_variable])],
        axis=1)
    logger.debug(f'After undersample')
    describe(aux, target_variable)

    # apply over-sampling to minority classes
    oversampled = RandomOverSampler(sampling_strategy=oversample_dict, random_state=42)
    X_sampled, y_sampled = oversampled.fit_resample(X_undersampled, y_undersampled)
    aux2 = pd.concat(
        [pd.DataFrame(X_sampled), pd.DataFrame(y_sampled, columns=[target_variable])],
        axis=1)
    logger.debug(f'After oversample')
    describe(aux2, target_variable)

    # apply SMOTE to balance the dataset
    smote = SMOTE(sampling_strategy=get_smote_dictionary(aux2, target_variable), random_state=42)
    X_undersampled_resampled, y_undersampled_resampled = smote.fit_resample(X_sampled, y_sampled)

    # create a new dataframe with the resampled and under-sampled data
    return pd.concat(
        [pd.DataFrame(X_undersampled_resampled), pd.DataFrame(y_undersampled_resampled, columns=[target_variable])],
        axis=1)


def get_smote_dictionary(df: pd.DataFrame, target_class: str) -> dict:
    """
    This function calculates a dynamic dictionary for Oversampling a dataset with the SMOTE technique.
    Args:
        df: Dataframe to be oversampled.
        target_class: Class that is going to be oversampled.

    Returns: An optimal relation of values for oversampling a dataset without losing too much precision.

    """
    logger = logging.getLogger('ThreatTrekker')
    logger.debug('get_smote_dictionary')

    # Return the number of samples per class
    class_value_counts = df[target_class].value_counts(normalize=False)
    mean = class_value_counts.mean()
    max = class_value_counts.max()
    ret = {}
    logger.debug(f'mean: {mean}, max: {max}')
    for class_val in df[target_class].unique():
        values = class_value_counts[class_val]
        ret[class_val] = values
        if values + math.sqrt(mean) < mean:
            ret[class_val] += math.ceil(math.sqrt(max))

    logger.debug(f'SMOTE dict: {ret}')
    return ret
