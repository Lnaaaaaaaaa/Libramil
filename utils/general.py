import os
import re
import csv
import yaml
import json
import glob
import shutil
import random
import numpy as np
from pathlib import Path

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import label_binarize
from typing import List, Tuple, Union

def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_score: Union[List[float], List[List[float]], np.ndarray],
    num_classes: int = 2
) -> Tuple[float, float, float]:
    
    acc = accuracy_score(y_true, y_pred)

    y_score = np.array(y_score)

    if num_classes == 2:
        # 处理形状为 (N, 2) 的情况，取正类概率
        if y_score.ndim == 2 and y_score.shape[1] == 2:
            pos_probs = y_score[:, 1]
        else:
            pos_probs = y_score
        auc = roc_auc_score(y_true, pos_probs)
    else:
        # 多分类需要 binarize 标签
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        auc = roc_auc_score(y_true_bin, y_score, multi_class='ovr')

    average_mode = 'binary' if num_classes == 2 else 'macro'
    f1 = f1_score(y_true, y_pred, average=average_mode)

    return acc, auc, f1

class CSVWriter:
    def __init__(self, filename, header=None, sep=',', append=False):
        self.filename = filename
        self.sep = sep
        if Path(self.filename).exists() and not append:
            os.remove(self.filename)
        if header is not None:
            self.write_row(header)

    def write_row(self, row):
        with open(self.filename, 'a+') as fp:
            csv_writer = csv.writer(fp, delimiter=self.sep)
            csv_writer.writerow(row)
            fp.flush()

    def write_rows(self, rows):
        with open(self.filename, 'a+') as fp:
            csv_writer = csv.writer(fp, delimiter=self.sep)
            csv_writer.writerows(rows)
            fp.flush()

def write_summary_log(final_log_path, summary_log_path=None):
    if summary_log_path is None:
        summary_log_path = os.path.join(os.path.dirname(final_log_path), 'summary_log.csv')

    df = pd.read_csv(final_log_path)
    df = df[pd.to_numeric(df['flod'], errors='coerce').notnull()]

    mean_vals = df[['test_acc', 'test_auc', 'test_f1_score']].mean()
    std_vals = df[['test_acc', 'test_auc', 'test_f1_score']].std()

    summary_df = pd.DataFrame([
        ['mean'] + mean_vals.round(4).tolist(),
        ['std'] + std_vals.round(4).tolist()
    ], columns=['metric', 'test_acc', 'test_auc', 'test_f1_score'])

    summary_df.to_csv(summary_log_path, index=False)
