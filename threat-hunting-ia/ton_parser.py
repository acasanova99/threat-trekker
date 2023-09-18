"""
    ton_parser.py

    This module contains all the necessary functions to build, encode, and process the ton-dataset.

    Author: Angel Casanova
    2023
"""
import logging
from pathlib import Path
from typing import Final
import math
import pandas as pd
import pyarrow.parquet as pq
from ml_encoder import *
from common_parser import CommonParser


class TonParser(CommonParser):
    def __init__(self):
        super().__init__('csv', 'type', 'ton-dataset')

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

        # Remove some columns (almost all rows were empty for these columns)
        remove_column(df, 'service')
        remove_column(df, 'missed_bytes')
        remove_column(df, 'dns_query')
        remove_column(df, 'dns_qclass')
        remove_column(df, 'dns_qtype')
        remove_column(df, 'dns_rcode')
        remove_column(df, 'dns_aa')
        remove_column(df, 'dns_rd')
        remove_column(df, 'dns_ra')
        remove_column(df, 'dns_rejected')
        remove_column(df, 'ssl_version')
        remove_column(df, 'ssl_cipher')
        remove_column(df, 'ssl_resumed')
        remove_column(df, 'ssl_established')
        remove_column(df, 'ssl_subject')
        remove_column(df, 'ssl_issuer')
        remove_column(df, 'http_trans_depth')
        remove_column(df, 'http_method')
        remove_column(df, 'http_uri')
        remove_column(df, 'http_version')
        remove_column(df, 'http_request_body_len')
        remove_column(df, 'http_response_body_len')
        remove_column(df, 'http_status_code')
        remove_column(df, 'http_user_agent')
        remove_column(df, 'http_orig_mime_types')
        remove_column(df, 'http_resp_mime_types')
        remove_column(df, 'weird_name')
        remove_column(df, 'weird_addl')
        remove_column(df, 'weird_notice')
        remove_column(df, 'label')

        # Encode Ipv4 addressesL
        ipv4_to_int32(df, 'src_ip')
        ipv4_to_int32(df, 'dst_ip')

        # Encode categorical values
        super()._print_column_info(df, 'proto')
        df = one_hot_for_categorical(df, 'proto')

        # Encode categorical values
        super()._print_column_info(df, 'conn_state')
        df = one_hot_for_categorical(df, 'conn_state')

        #  Output Class
        super()._print_column_info(df, 'type')
        df, le = label_encoder(df, 'type')

        return df, le

    def get_undersample_dictionary(self, df: pd.DataFrame) -> dict:
        # As there is a huge difference among all the samples, for this dataset, the factor needs to be also high
        majority_factor = 0.8

        # Return the number of samples per class
        class_value_counts = df[self.label].value_counts(normalize=False)
        return {
            0: math.ceil(class_value_counts[0] * majority_factor)
        }

    def get_oversample_dictionary(self, df: pd.DataFrame) -> dict:
        minority_factor = 1.2

        # Return the number of samples per class
        class_value_counts = df[self.label].value_counts(normalize=False)
        return {
            1: math.ceil(class_value_counts[1] * minority_factor)
        }

    def label_dictionary(self) -> dict:
        return {
            0: "Backdoor",
            1: "Ddos",
            2: "DoS",
            3: "Injection",
            4: "Mitm",
            5: "Benign",
            6: "Passwd",
            7: "Ransom.",
            8: "Scan.",
            9: "Xss",
        }
