import random

import numpy as np
import torch


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def cal_metrics(cf_matrix):
    n_classes = cf_matrix.shape[0]
    metrics_result = []
    count_record = []
    for i in range(n_classes):

        ALL = np.sum(cf_matrix)
        # The diagonal is correctly predicted
        TP = cf_matrix[i, i]
        FP = np.sum(cf_matrix[:, i]) - TP
        FN = np.sum(cf_matrix[i, :]) - TP
        TN = ALL - TP - FP - FN
        # 0 Precision
        # 1 Sensitivity=recall
        # 2 Specificity
        # 3 accuracy
        # 4 f1_score
        precision = (TP / (TP + FP))
        sensitivity = (TP / (TP + FN))
        specificity = (TN / (TN + FP))
        accuracy = (TN + TP) / ALL
        if TP + FP == 0:
            precision = 0
        if TP + FN == 0:
            sensitivity = 0
        if TN + FP == 0:
            specificity = 0
        if ALL == 0:
            accuracy = 0
        if precision == 0 or sensitivity == 0:
            f1_score = 0
        else:
            f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
        metrics_result.append([precision, sensitivity, specificity, accuracy, f1_score])
        count_record.append([TP, FP, FN, TN])
    return metrics_result, count_record
