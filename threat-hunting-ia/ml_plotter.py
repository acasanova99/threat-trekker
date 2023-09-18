"""
    ml_plotter.py

    This module contains all the functionality to plot the result of the ML analysis in a more human-readable way.

    Author: Angel Casanova
    2023
"""
import logging
from typing import Final

import numpy as np
import pandas as pd

from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

PLOTS_PATH: Final[str] = './data/plots/'


def generate_roc_curve(y, y_pred):
    """ Generates a roc_curve plot based on the given values.
    Args:
        y (any): the resulting y_test of the function "train_test_split".
        y_prob (any): the resulting value of the function "predict".
    Returns:
        _.
    """
    logger = logging.getLogger('ThreatTrekker')
    logger.debug('Plotting ROC Curve')
    logger.info(f'y_prob')
    logger.info(f'y_prob[:, 1]')

    # calculate the false positive rate, true positive rate, and thresholds
    fpr, tpr, thresholds = roc_curve(y, y_pred[:, 1], pos_label=1)

    # plot the ROC curve using matplotlib
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig(PLOTS_PATH + 'ROC Curve')


def generate_confusion_matrix(y_test, y_pred, parser, cm_ori=None):
    """ Generates a confusion_matrix plot based on the given values.
    Args:
        y_test (any): the resulting y_test of the function "train_test_split".
        y_pred (any): the resulting value of the function "predict".
    Returns:
        _.
    """
    logger = logging.getLogger('ThreatTrekker')
    logger.debug('Plotting confusion matrix')
    if cm_ori is None:
        cm = confusion_matrix(y_test, y_pred)
        return cm
    else:
        cm = cm_ori
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Get the class names from the static parser
    class_names_dict = parser.label_dictionary()
    class_names = [class_names_dict[i] for i in range(len(class_names_dict))]

    # plot the confusion matrix using seaborn
    sns.set(rc={'figure.figsize': (12, 10)})  # nothing for uwf, 10:6 for cic
    heatmap = sns.heatmap(cm_norm, annot=True, cmap='Blues', fmt='.2f', xticklabels=class_names,
                          yticklabels=class_names)

    # Rotate x-axis labels by 45 degrees for better fit
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha='right')
    # heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, ha='right')

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(PLOTS_PATH + 'Confusion Matrix')
    # plt.show()
    logger.info(f'\n{cm_norm}')


def generate_barchart(clf, x):
    """ Generates a confusion_matrix plot based on the given values.
    Args:
        clf (any): the classifier that has been used.
        x (any): the resulting x of the function "train_test_split".
    Returns:
        _.
    """
    logger = logging.getLogger('ThreatTrekker')
    logger.debug(f'Plotting bar chart')

    feature_names = [f"feature {i}" for i in range(x.shape[1])]
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)

    forest_importances = pd.Series(importances, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.savefig(PLOTS_PATH + 'Barchart')
