"""
    cic_parser.py

    This module contains all the necessary functions to build, encode, and process the cic-dataset.

    Author: Angel Casanova
    2023
"""
from typing import Tuple, Any

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ml_encoder import *
from common_parser import CommonParser
from pathlib import Path


class CicParser(CommonParser):
    def __init__(self):
        super().__init__('csv', 'label', 'cic-dataset')

    def build_from_dataset(self) -> pd.DataFrame:
        data_dir: Path = Path(CommonParser.DATA_PATH + self.name + '/')
        df = pd.concat(
            pd.read_csv(input_file)
            for input_file in data_dir.glob('*.' + self.legacy_extension)
        )

        df.columns = df.columns.str.strip().str.lower() \
            .str.replace(' ', '_', regex=False) \
            .str.replace('(', '', regex=False) \
            .str.replace(')', '', regex=False)
        return df

    def preprocess(self, df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
        le: LabelEncoder

        # Remove some columns
        remove_column(df, 'idle_min')
        remove_column(df, 'idle_max')
        remove_column(df, 'idle_std')
        remove_column(df, 'idle_mean')
        remove_column(df, 'active_min')
        remove_column(df, 'active_max')
        remove_column(df, 'active_std')
        remove_column(df, 'active_mean')
        remove_column(df, 'bwd_avg_bulk_rate')
        remove_column(df, 'bwd_avg_packets/bulk')
        remove_column(df, 'bwd_avg_bytes/bulk')
        remove_column(df, 'fwd_avg_bulk_rate')
        remove_column(df, 'fwd_avg_packets/bulk')
        remove_column(df, 'fwd_avg_bytes/bulk')
        remove_column(df, 'down/up_ratio')
        remove_column(df, 'ece_flag_count')
        remove_column(df, 'cwe_flag_count')
        remove_column(df, 'urg_flag_count')
        remove_column(df, 'ack_flag_count')
        remove_column(df, 'psh_flag_count')
        remove_column(df, 'rst_flag_count')
        remove_column(df, 'syn_flag_count')
        remove_column(df, 'fin_flag_count')
        remove_column(df, 'bwd_urg_flags')
        remove_column(df, 'fwd_urg_flags')
        remove_column(df, 'bwd_psh_flags')
        remove_column(df, 'fwd_psh_flags')
        remove_column(df, 'bwd_iat_min')
        remove_column(df, 'bwd_iat_max')
        remove_column(df, 'bwd_iat_std')
        remove_column(df, 'bwd_iat_mean')
        remove_column(df, 'bwd_iat_total')
        remove_column(df, 'fwd_iat_min')
        remove_column(df, 'fwd_iat_max')
        remove_column(df, 'fwd_iat_std')
        remove_column(df, 'fwd_iat_mean')
        remove_column(df, 'fwd_iat_total')
        remove_column(df, 'flow_iat_min')
        remove_column(df, 'flow_iat_max')
        remove_column(df, 'flow_iat_std')
        remove_column(df, 'flow_iat_mean')
        remove_column(df, 'bwd_packet_length_std')
        remove_column(df, 'bwd_packet_length_mean')
        remove_column(df, 'bwd_packet_length_min')
        remove_column(df, 'bwd_packet_length_max')
        remove_column(df, 'fwd_packet_length_std')
        remove_column(df, 'fwd_packet_length_mean')
        remove_column(df, 'fwd_packet_length_min')
        remove_column(df, 'fwd_packet_length_max')

        remove_column(df, 'packet_length_mean')
        remove_column(df, 'packet_length_std')
        remove_column(df, 'packet_length_variance')
        remove_column(df, 'average_packet_size')
        remove_column(df, 'subflow_fwd_packets')
        remove_column(df, 'subflow_fwd_bytes')
        remove_column(df, 'subflow_bwd_packets')
        remove_column(df, 'subflow_bwd_bytes')
        remove_column(df, 'fwd_header_length.1')

        CommonParser._print_column_info(df, 'label')
        df, le = label_encoder(df, 'label')

        return df, le

    def get_undersample_dictionary(self, df: pd.DataFrame) -> dict:
        # As there is a huge difference among all the samples, for this dataset, the factor needs to be also high
        majority_factor1 = 50
        majority_factor2 = 10

        # Return the number of samples per class
        class_value_counts = df[self.label].value_counts(normalize=False)
        return {
            0: class_value_counts[0] // majority_factor1,
            4: class_value_counts[4] // majority_factor2,
            10: class_value_counts[10] // majority_factor2,
            2: class_value_counts[2] // majority_factor2,
        }

    def get_oversample_dictionary(self, df: pd.DataFrame) -> dict:
        minority_factor1 = 10
        minority_factor2 = 2

        # Return the number of samples per class
        class_value_counts = df[self.label].value_counts(normalize=False)
        return {
            8: class_value_counts[8] * minority_factor1,
            9: class_value_counts[9] * minority_factor1,
            13: class_value_counts[13] * minority_factor1,
            14: class_value_counts[14] * minority_factor2,
        }

    def label_dictionary(self) -> dict:
        return {
            0: "Benign",
            1: "Bot",
            2: "DDoS",
            3: "GoldenEye",
            4: "DoS Hulk",
            5: "Slowhttptest",
            6: "Slowloris",
            7: "FTP-Patator",
            8: "Heartbleed",
            9: "Infiltration",
            10: "PortScan",
            11: "SSH-Patator",
            12: "Brute Force",
            13: "SqlI",
            14: "Web XSS",
        }
