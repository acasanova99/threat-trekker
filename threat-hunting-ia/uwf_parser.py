"""
    uwf_parser.py

    This module contains all the necessary functions to build, encode, and process the uwf-dataset.

    Author: Angel Casanova
    2023
"""
import logging
from typing import Final, Tuple

import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder

from ml_encoder import *
from common_parser import CommonParser


class UwfParser(CommonParser):
    def __init__(self):
        super().__init__('parquet', 'label_tactic', 'uwf-dataset')

    def preprocess(self, df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
        le: LabelEncoder

        # Remove some columns
        remove_column(df, 'history')
        remove_column(df, 'community_id')
        remove_column(df, 'uid')

        # Encode Ipv4 addressesL
        ipv4_to_int32(df, 'dest_ip_zeek')
        ipv4_to_int32(df, 'src_ip_zeek')

        # Encode bool values
        df = yorn(df, 'local_orig')

        # Encode categorical values
        super()._print_column_info(df, 'service')
        df = one_hot_for_categorical(df, 'service')

        super()._print_column_info(df, 'proto')
        df = one_hot_for_categorical(df, 'proto')

        super()._print_column_info(df, 'conn_state')
        df = one_hot_for_categorical(df, 'conn_state')

        super()._print_column_info(df, 'label_tactic')
        df, le = label_encoder(df, 'label_tactic')

        # Encode date values
        df = ts_to_epoch(df, 'datetime')

        return df, le

    def remove_unbalance_classes(self, df: pd.DataFrame, balanced: bool = False):
        """ As the uwf-dataset is highly unbalance, this method removes the less representative classes of the dataset.
        Args:
            df (pandas.DataFrame): data frame that is going to be trimmed.
            balanced(bool): indicated if the dataframe id going to be balanced or not.
        Returns:
            _
        """
        if balanced:
            return df[(df.label_tactic != 6) & (df.label_tactic != 4) & (df.label_tactic != 1)]
        else:
            return df[(df.label_tactic != 6) & (df.label_tactic != 4) & (df.label_tactic != 1)
                      & (df.label_tactic != 5) & (df.label_tactic != 9) & (df.label_tactic != 3)]

    def get_undersample_dictionary(self, df: pd.DataFrame) -> dict:
        # As there is a huge difference among all the samples, for this dataset, the factor needs to be also high
        majority_factor = 1000

        # Return the number of samples per class
        class_value_counts = df[self.label].value_counts(normalize=False)
        return {
            10: class_value_counts[10] // majority_factor,
            8: class_value_counts[8] // majority_factor
        }

    def get_oversample_dictionary(self, df: pd.DataFrame) -> dict:
        minority_factor = 10

        # Return the number of samples per class
        class_value_counts = df[self.label].value_counts(normalize=False)
        return {
            0: class_value_counts[0] * minority_factor,
            7: class_value_counts[7] * minority_factor,
            3: class_value_counts[3] * minority_factor,
            5: class_value_counts[5] * minority_factor,
            9: class_value_counts[9] * minority_factor,
        }

    def label_dictionary(self) -> dict:
        return {
            0: "Cred. Access",
            1: "Discovery",
            2: "Exfil.",
            3: "Lat. Movement",
            4: "Priv. Escal.",
            5: "Reconn.",
            6: "Res. Develop.",
            7: "Benign",
        }
