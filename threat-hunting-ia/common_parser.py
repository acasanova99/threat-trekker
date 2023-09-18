"""
    common_parser.py

    This module contains all the necessary signatures and common functions to build, encode, and process the dataset
    that are going to be processed by the Threat Trekker algorithm.

    Author: Angel Casanova
    2023
"""
from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Final

import pandas as pd
import pyarrow.parquet as pq
from ml_encoder import *


class CommonParser(ABC):
    DATA_PATH: Final[str] = './data/datasets/'
    PARSED_PATH: Final[str] = '/parsed/'

    label: str = None
    legacy_extension: str = None
    name: str = None

    @abstractmethod
    def __init__(self, legacy_extension: str, label: str, name: str):
        self.legacy_extension = legacy_extension
        self.label = label
        self.name = name

    @staticmethod
    def _print_column_info(df: pd.DataFrame, column_name: str) -> None:
        """ Prints the most representative information about one column of the dataset.
        Args:
            df (pandas.DataFrame): data frame that is going to be described.
            column_name (str): name of the column that is going to be described.
        Returns:
            _
        """
        logger = logging.getLogger('ThreatTrekker')
        logger.info(f"\n\n {'*' * 10} {column_name} {'*' * 10}")
        logger.info(f'nunique(): {str(df[column_name].nunique())}')
        logger.info(f'unique(): {str(df[column_name].unique())}')
        logger.info(f'value_counts\n {str(df[column_name].value_counts())}')
        logger.info(f'list n-first values\n {str(df[column_name].values)}')
        return

    def build_from_dataset(self) -> pd.DataFrame:
        """ Builds up a Pandas pd.DataFrame with all the files within the given directory.
        Returns:
            _ (pandas.DataFrame): A DataFrame that gathers the whole given dataset.
        """

        data_dir: Path = Path(CommonParser.DATA_PATH + self.name + '/')
        return pd.concat(
            pd.read_parquet(input_file)
            for input_file in data_dir.glob('*.' + self.legacy_extension)
        )

    def sample_from_dataset(self, dataset) -> pd.DataFrame:
        """ Builds up a Pandas Dataframe with the first file found within the given directory. This function is useful to
            load a sample of a dataset.
        Args:
            dataset (str): name of the dataset to be loaded within the DATA_PATH directory.
        Returns:
            _ (pandas.DataFrame): A DataFrame consisting of a single file of the desired dataset.
        """

        file_lst = Path(CommonParser.DATA_PATH + dataset).glob('*.' + self.legacy_extension)
        first_file = [f for f in file_lst][0]
        return pq.read_table(first_file).to_pandas()

    def store_df_as_parquet(self, df: pd.DataFrame, file_name: str) -> None:
        """ Stores a complete pandas' Dataframe into a single .parquet file within the DATA_PATH directory.
         This function is useful to keep in disk a preprocessed dataset.
        Args:
            df (pandas.Dataframe): dataframe that is going to be persisted.
            file_name (str): name of the file to be stored.
        Returns:
            _
        """

        df.to_parquet(CommonParser.DATA_PATH + self.name + CommonParser.PARSED_PATH + file_name)
        return

    def load_parquet_as_df(self, path: str, file_name: str) -> pd.DataFrame:
        """ Load a single .parquet file within the DATA_PATH directory as a Pandas DataFrame.
         This function is useful to keep in disk a preprocessed dataset.
        Args:
            path (str): that that is going to be concatenated to DATA_PATH. It works as a dataset selector.
            file_name (str): name of the file to be loaded.
        Returns:
            _ (pandas.Dataframe): dataframe that is going to be loaded.
        """
        file = next(Path(CommonParser.DATA_PATH + path).glob(file_name))
        return pq.read_table(file).to_pandas()

    def remove_unbalance_classes(self, df: pd.DataFrame, balanced: bool = False) -> pd.DataFrame:
        """ As some datasets are highly unbalance, this method removes the less representative classes of the dataset.
        Args:
            df (pandas.DataFrame): data frame that is going to be trimmed.
            balanced(bool): indicated if the dataframe is going to be balanced or not.
        Returns:
            _
        """
        return df  # Do nothing by default

    def label_dictionary(self) -> dict:
        """ This method stores static information about the labels of the dataset.

        Returns:
            dict (dict): The dictionary with the Labels for the plot
        """
        return {
            0: "Benign",
            1: "Malicious",
        }

    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
        """ Preprocess the dataset to follow the standard of this project.
        Args:
            df (pandas.Dataframe): Dataframe that is going to be processed.
        Returns:
            tuple (pandas.Dataframe, LabelEncoder): The encoded Dataframe and the Label Encoder.
        """
        pass

    @abstractmethod
    def get_undersample_dictionary(self, df: pd.DataFrame) -> dict:
        """ As some datasets are highly unbalance, this method removes some samples of the majority classes.
        Args:
            df (pandas.DataFrame): data frame that is going to be trimmed.
        Returns:
            dict (dict): The dictionary with the number of samples per class that will be removed.
        """
        pass

    @abstractmethod
    def get_oversample_dictionary(self, df: pd.DataFrame) -> dict:
        """ As some datasets are highly unbalance, this method adds some samples to the minority classes.
        Args:
            df (pandas.DataFrame): data frame that is going to be trimmed.
        Returns:
            dict (dict): The dictionary with the number of samples per class that will be added.
        """
        pass
