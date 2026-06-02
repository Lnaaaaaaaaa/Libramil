"""
Libra-MIL 工具函数模块
======================
包含模型训练和评估所需的通用工具函数：
- compute_metrics: 计算分类指标（准确率、AUC、F1分数）
- CSVWriter: CSV日志写入器
- write_summary_log: 汇总多折交叉验证结果
"""

import os
import re
import csv
import yaml
import json
import glob
import shutil
import random
import numpy as np
import pandas as pd
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
    """
    计算分类任务的评估指标

    参数:
        y_true: 真实标签列表，shape: (N,)
        y_pred: 预测标签列表，shape: (N,)
        y_score: 预测概率/分数，shape: (N,) 或 (N, num_classes)
                 二分类时取正类概率，多分类时为各类别概率
        num_classes: 类别数量，默认为2（二分类）

    返回:
        acc: 准确率 (Accuracy)
        auc: ROC曲线下面积 (Area Under Curve)
        f1: F1分数 (精确率和召回率的调和平均)

    说明:
        - 二分类: AUC计算使用正类概率
        - 多分类: AUC使用one-vs-rest策略，F1使用macro平均
    """
    # 计算准确率：预测正确的比例
    acc = accuracy_score(y_true, y_pred)

    y_score = np.array(y_score)

    if num_classes == 2:
        # ========== 二分类情况 ==========
        # 处理形状为 (N, 2) 的情况，取正类概率（第二列）
        if y_score.ndim == 2 and y_score.shape[1] == 2:
            pos_probs = y_score[:, 1]  # 取正类概率
        else:
            pos_probs = y_score
        # 计算二分类AUC
        auc = roc_auc_score(y_true, pos_probs)
    else:
        # ========== 多分类情况 ==========
        # 将标签转为one-hot编码: shape (N, num_classes)
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        # 使用one-vs-rest策略计算多分类AUC
        auc = roc_auc_score(y_true_bin, y_score, multi_class='ovr')

    # 计算F1分数：二分类用binary，多分类用macro平均
    average_mode = 'binary' if num_classes == 2 else 'macro'
    f1 = f1_score(y_true, y_pred, average=average_mode)

    return acc, auc, f1


class CSVWriter:
    """
    CSV日志写入器
    =============
    用于记录训练过程中的指标和结果

    使用示例:
        writer = CSVWriter('log.csv', header=['epoch', 'loss', 'acc'])
        writer.write_row([1, 0.5, 0.8])
        writer.write_rows([[2, 0.4, 0.85], [3, 0.3, 0.9]])
    """

    def __init__(self, filename, header=None, sep=',', append=False):
        """
        初始化CSV写入器

        参数:
            filename: CSV文件路径
            header: 表头列表，如 ['epoch', 'loss', 'acc']
            sep: 分隔符，默认为逗号
            append: 是否追加模式，False时会覆盖已存在的文件
        """
        self.filename = filename
        self.sep = sep

        # 如果文件已存在且不是追加模式，则删除旧文件
        if Path(self.filename).exists() and not append:
            os.remove(self.filename)

        # 写入表头
        if header is not None:
            self.write_row(header)

    def write_row(self, row):
        """写入单行数据"""
        with open(self.filename, 'a+') as fp:
            csv_writer = csv.writer(fp, delimiter=self.sep)
            csv_writer.writerow(row)
            fp.flush()  # 立即刷新到磁盘，防止程序崩溃丢失数据

    def write_rows(self, rows):
        """写入多行数据"""
        with open(self.filename, 'a+') as fp:
            csv_writer = csv.writer(fp, delimiter=self.sep)
            csv_writer.writerows(rows)
            fp.flush()


def write_summary_log(final_log_path, summary_log_path=None):
    """
    汇总多折交叉验证的结果

    参数:
        final_log_path: 各折结果的CSV文件路径
        summary_log_path: 汇总结果保存路径，默认保存在同目录下

    功能:
        读取各折的测试结果，计算均值和标准差，生成汇总报告
    """
    if summary_log_path is None:
        summary_log_path = os.path.join(os.path.dirname(final_log_path), 'summary_log.csv')

    # 读取结果文件
    df = pd.read_csv(final_log_path)
    # 过滤掉非数值行（防止表头等干扰）
    df = df[pd.to_numeric(df['flod'], errors='coerce').notnull()]

    # 计算各指标的均值和标准差
    mean_vals = df[['test_acc', 'test_auc', 'test_f1_score']].mean()
    std_vals = df[['test_acc', 'test_auc', 'test_f1_score']].std()

    # 构建汇总DataFrame
    summary_df = pd.DataFrame([
        ['mean'] + mean_vals.round(4).tolist(),
        ['std'] + std_vals.round(4).tolist()
    ], columns=['metric', 'test_acc', 'test_auc', 'test_f1_score'])

    summary_df.to_csv(summary_log_path, index=False)
