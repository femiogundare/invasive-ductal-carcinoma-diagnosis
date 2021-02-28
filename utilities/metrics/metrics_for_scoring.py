# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 21:35:47 2021

@author: femiogundare
"""


# Performance metrics
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
"""
def optimal_threshold(y_true, y_prob):
    # Returns the optimal threshold based on the false positive rate, true positive rate, and thresholds.
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    opt_i = np.argmax(tpr - fpr)
    return thresholds[opt_i]
"""

def optimal_threshold(y_true, y_prob):
    # Returns the optimal threshold based on the false positive rate, true positive rate, and thresholds.
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    J = tpr - fpr
    ix = np.argmax(J)
    opt_i = thresholds[ix]
    return opt_i

def optimal_conf_matrix(y_true, y_prob):
    # Returns the optimal confusion matrix based on the optimal threshold.
    c = confusion_matrix(y_true, (y_prob > optimal_threshold(y_true, y_prob))*1)
    return c

def opt_sensitivity_score(y_true, y_prob):
    # Returns the optimal sensitivity score based on the optimal threshold.
    c = optimal_conf_matrix(y_true, y_prob)
    return c[1][1]/(c[1][1] + c[1][0])

def opt_specificity_score(y_true, y_prob):
    # Returns the optimal specificity score based on the optimal threshold.
    c = optimal_conf_matrix(y_true, y_prob)
    return c[0][0]/(c[0][0] + c[0][1])

def opt_ppv_score(y_true, y_prob):
    # Returns the optimal ppv score based on the optimal threshold.
    c = optimal_conf_matrix(y_true, y_prob)
    return c[1][1]/(c[1][1] + c[0][1])

def opt_npv_score(y_true, y_prob):
    # Returns the optimal npv score based on the optimal threshold.
    c = optimal_conf_matrix(y_true, y_prob)
    return c[0][0]/(c[0][0] + c[1][0])

def opt_J_score(y_true, y_prob):
    # Returns the optimal specificity score based on the optimal threshold.
    sensitivity = opt_sensitivity_score(y_true, y_prob)
    specificity = opt_specificity_score(y_true, y_prob)
    return (sensitivity + specificity - 1)

def opt_auc_score(y_true, y_prob):
    # Returns the optimal AUC score based on the optimal threshold.
    opt_t = optimal_threshold(y_true, y_prob)
    y_pred = (y_prob > opt_t)*1
    return roc_auc_score(y_true, y_pred)

def opt_threshold_score(y_true, y_prob):
    return optimal_threshold(y_true, y_prob)