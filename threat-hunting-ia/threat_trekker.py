#!/usr/bin/env python
import math
import subprocess
import sys

from argparse import *
import logging
import joblib
import numpy as np
from typing import Final

import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

import ml_encoder
import ml_plotter
from simple_logger import CustomFormatter
from parser_factory import ParserFactory
from common_parser import CommonParser

BANNER_80: Final[str] = """
 ███████████ █████                               █████                   
░█░░░███░░░█░░███                               ░░███                    
░   ░███  ░  ░███████  ████████ ██████  ██████  ███████                  
    ░███     ░███░░███░░███░░█████░░███░░░░░███░░░███░                   
    ░███     ░███ ░███ ░███ ░░░███████  ███████  ░███                    
    ░███     ░███ ░███ ░███   ░███░░░  ███░░███  ░███ ███                
    █████    ████ ██████████  ░░██████░░████████ ░░█████                 
   ░░░░░    ░░░░ ░░░░░░░░░░    ░░░░░░  ░░░░░░░░   ░░░░░                  
              ███████████               █████     █████                        
             ░█░░░███░░░█              ░░███     ░░███                         
             ░   ░███  ░████████ ██████ ░███ █████░███ █████  ██████  ████████ 
                 ░███  ░░███░░█████░░███░███░░███ ░███░░███  ███░░███░░███░░███
                 ░███   ░███ ░░░███████ ░██████░  ░██████░  ░███████  ░███ ░░░ 
                 ░███   ░███   ░███░░░  ░███░░███ ░███░░███ ░███░░░   ░███     
                 █████  █████  ░░██████ ████ █████████ █████░░██████  █████    
                ░░░░░  ░░░░░    ░░░░░░ ░░░░ ░░░░░░░░░ ░░░░░  ░░░░░░  ░░░░░                                                                                                                                     
"""
BANNER: Final[str] = """
 ███████████ █████                               █████  ███████████               █████     █████                        
░█░░░███░░░█░░███                               ░░███  ░█░░░███░░░█              ░░███     ░░███                         
░   ░███  ░  ░███████  ████████ ██████  ██████  ███████░   ░███  ░████████ ██████ ░███ █████░███ █████  ██████  ████████ 
    ░███     ░███░░███░░███░░█████░░███░░░░░███░░░███░     ░███  ░░███░░█████░░███░███░░███ ░███░░███  ███░░███░░███░░███
    ░███     ░███ ░███ ░███ ░░░███████  ███████  ░███      ░███   ░███ ░░░███████ ░██████░  ░██████░  ░███████  ░███ ░░░ 
    ░███     ░███ ░███ ░███   ░███░░░  ███░░███  ░███ ███  ░███   ░███   ░███░░░  ░███░░███ ░███░░███ ░███░░░   ░███     
    █████    ████ ██████████  ░░██████░░████████ ░░█████   █████  █████  ░░██████ ████ █████████ █████░░██████  █████    
   ░░░░░    ░░░░ ░░░░░░░░░░    ░░░░░░  ░░░░░░░░   ░░░░░   ░░░░░  ░░░░░    ░░░░░░ ░░░░ ░░░░░░░░░ ░░░░░  ░░░░░░  ░░░░░                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
"""
BANNER_LEN: Final[int] = 121
BANNER_80_LEN: Final[int] = 80
CENTER_MADE_BY: Final[float] = .4
MADE_BY: Final[str] = 'Made by: A. Casanova\n'
USE_ALL_THREADS: Final[int] = -1
MODELS_PATH: Final[str] = './data/models/'


def main(argv: Namespace, parser: CommonParser):
    logger = logging.getLogger('ThreatTrekker')

    # Loads a dataframe
    split_path: list[str] = argv.input_dataset.split('/')  # Split the file from the path
    df: pd.DataFrame = parser.load_parquet_as_df('/'.join(split_path[:-1]), split_path[-1])

    logger.info(f'Dataframe dtypes (Initial):\n{df.dtypes}')
    ml_encoder.describe(df, parser.label)

    # Clean up the dataframe
    logger.info(f'rows before clean up {len(df.index)}')
    df = ml_encoder.clean_dataset(df)
    logger.info(f'rows after clean up {len(df.index)}')

    # Balance the dataset if needed
    if argv.balance:
        logger.info(f'Balancing the dataset')
        df = parser.remove_unbalance_classes(df, True)
        ml_encoder.describe(df, parser.label)

        df = ml_encoder.balance(df, parser.label,
                                parser.get_undersample_dictionary(df),
                                parser.get_oversample_dictionary(df))
    else:
        logger.info(f'Continue without balancing the dataset')
        df = parser.remove_unbalance_classes(df, False)

    logger.info(f'Dataframe dtypes (After Clean Up):\n{df.dtypes}')
    ml_encoder.describe(df, parser.label)

    # Store the processed dataframe into a file
    if argv.out_dataset is not None:
        parser.store_df_as_parquet(df, argv.out_dataset)

    # Take a sample, otherwise may be too heavy
    logger.info(f'Sampling the Data')
    df = df.sample(frac=argv.frac)
    logger.info(f'rows after sampling: {len(df.index)}')

    x: np.array = df.loc[:, df.columns != parser.label].values
    y: np.array = df.loc[:, df.columns == parser.label].values.ravel()

    evaluation_results = []
    confusion_matrix_results = []

    for _ in range(argv.iters):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

        # If it is necessary to train the ML
        if argv.file_to_load is None:
            clf = RandomForestClassifier(
                verbose=1,
                n_jobs=USE_ALL_THREADS,
                max_depth=math.ceil(100 * argv.frac)
            )
            logger.info(f'Fitting Data')
            clf.fit(x_train, y_train)

            # Store the processed model into a file
            if argv.file_to_save is not None:
                joblib.dump(clf, MODELS_PATH + argv.file_to_save + '.joblib')
        else:
            clf = joblib.load(MODELS_PATH + argv.file_to_load + '.joblib')

        logger.info(f'Predicting data probs')
        y_predicted = clf.predict_proba(x_test)
        cls_loss = log_loss(y_test, y_predicted)
        logger.info(f'Log loss: {cls_loss}')

        logger.info(f'Predicting data classes')
        y_pred = clf.predict(x_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        logger.info(f'accuracy: {accuracy}')

        # Append the result to the list
        evaluation_results.append(accuracy)
        confusion_matrix_results.append(ml_plotter.generate_confusion_matrix(y_test, y_pred, parser))

    # Calculate the average of the evaluation results
    average_accuracy = np.mean(evaluation_results)
    average_confusion_matrix = np.mean(confusion_matrix_results, axis=0)
    logger.info(f'Total accuracy: {average_accuracy}')

    clf_rpt = classification_report(y_test, y_pred, zero_division=1)
    logger.info(f'{clf_rpt}')

    ml_plotter.generate_confusion_matrix(y_test, y_pred, parser, average_confusion_matrix)

    logger.info('Exiting thread-hunting-ia script')


def build_data_frame_from(out_file: str, skip_preprocess: bool, parser: CommonParser):
    logger = logging.getLogger('ThreatTrekker')
    logger.info(f'Building dataset from: {parser.name}')

    # Build a dataframe from stored dataset
    df: pd.DataFrame = parser.build_from_dataset()

    # Encode the input features to make them more handle for the ML engine
    if not skip_preprocess:
        logger.info(f'***** before preprocess: *****')
        logger.info(f'{df.columns}')
        logger.info(f'{df.dtypes}')
        df, le = parser.preprocess(df)
        logger.info(f'***** after preprocess: *****')
        logger.info(f'{df.columns}')
        logger.info(f'{df.dtypes}')

        # # Prnt the class relation:
        # aux = le.inverse_transform(le.classes_)
        #
        # for item1, item2 in zip(aux, le.classes_):
        #     logger.info(f'{item1} {item2} \n')

    # Store the processed dataframe into a file
    if out_file is not None:
        parser.store_df_as_parquet(df, out_file)


def likelihood_type(x):
    x = float(x)
    if x <= 0:
        raise ArgumentTypeError("The minimum sample space is is 0.1")
    elif x > 1:
        raise ArgumentTypeError("The maximum sample space is is 1 (That is, all the dataset)")
    return x


def print_banner(print_stdout: bool = False) -> str:
    ret: str = BANNER
    ret += ' ' * int(BANNER_LEN * 0.40)
    try:
        if int(subprocess.check_output(['stty', 'size']).split()[1]) < BANNER_LEN:
            ret = BANNER_80
            ret += ' ' * int(BANNER_80_LEN * CENTER_MADE_BY)
    except subprocess.CalledProcessError:
        ret = BANNER_80
        ret += ' ' * int(BANNER_80_LEN * CENTER_MADE_BY)

    ret += MADE_BY + '\n'

    if print_stdout:
        print(ret)

    return ret


if __name__ == '__main__':

    # Configure the argument parser
    arg_parser = ArgumentParser(
        prog='threat_trekker.py',
        description=print_banner() + 'Script for analyzing datasets oriented to the threat hunting.\n',
        epilog='Made by A.Casanova',
        formatter_class=RawTextHelpFormatter
    )
    arg_parser.add_argument('--input', '-i', dest='input_dataset',
                            help='Dataset that is desired to analyze. It should be placed into the '
                                 '<project_root>/data/datasets/ directory and it is needed to provided'
                                 ' datasets in the .parquet format',
                            required=True)
    arg_parser.add_argument('--output', '-o', dest='out_dataset',
                            help='File where the resulting dataset is going to be store after the preprocessing.'
                                 ' It would be placed under the <project_root>/data/datasets/ directory and it is needed to'
                                 ' be provided in the .parquet format',
                            required=False)
    arg_parser.add_argument('--skip-preprocess', '-s', dest='skip_preprocess', default=False,
                            help='Avoid the pre-processing of the data. It allows to try models with already encoded'
                                 ' datasets',
                            required=False, action='store_true')
    arg_parser.add_argument('--build-dataset', '-b', dest='build_dataset', default=False,
                            help='Build a dataset from all the .parquet files stored in the directory given as input.'
                                 ' It halts the execution after the preprocess of the data',
                            required=False, action='store_true')
    arg_parser.add_argument('--verbose', '-v', dest='encode', default=False,
                            help='Show the script traces on the current console',
                            required=False, action='store_true')
    arg_parser.add_argument('--sample', '-f', dest='frac', default=1.0, type=likelihood_type,
                            help='Percentage of the dataframe that is going to be used. Possible values: (0, 1]',
                            required=False)
    arg_parser.add_argument('--save-fit', dest='file_to_save',
                            help='Save the fitting model into a file, for using it in several predictions',
                            required=False)
    arg_parser.add_argument('--load-fit', dest='file_to_load',
                            help='Load the model to predict from a file, skipping all the training job',
                            required=False)
    arg_parser.add_argument('--balance', dest='balance', default=False,
                            help='Balance the input dataset before making the classification',
                            required=False, action='store_true')
    arg_parser.add_argument('--iters', dest='iters', default=1,
                            help='How many times do you want to train the model',
                            required=False, type=int)
    args: Namespace = arg_parser.parse_args()

    # Configures the logging
    CustomFormatter.setup_logging('ThreatTrekker')

    # Print the banner
    print_banner(True)

    # Generate the parser for the dataset
    cp: CommonParser = ParserFactory.instantiate_parser(args.input_dataset)

    # Execute the desired functionality
    if args.build_dataset:
        build_data_frame_from(args.out_dataset, args.skip_preprocess, cp)
    else:
        main(args, cp)

    sys.exit(0)
