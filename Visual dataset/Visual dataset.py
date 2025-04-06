# encoding: utf-8
"""
@author: wly
@software: PyCharm
@file: Visual dataset.py
@time: 2025/4/6 10:43
ETT-small 数据集介绍：
介绍及链接：https://github.com/zhouhaoyi/ETDataset/blob/main/README_CN.md
提供了两年的数据，每个数据点每分钟记录一次（用 m 标记），它们分别来自中国同一个省的两个不同地区，
分别名为ETT-small-m1和ETT-small-m2。
每个数据集包含2年 * 365天 * 24小时 * 4 = 70,080数据点。
此外，我们还提供一个小时级别粒度的数据集变体使用（用 h 标记），即ETT-small-h1和ETT-small-h2。
每个数据点均包含8维特征，包括数据点的记录日期、预测值“油温”以及6个不同类型的外部负载值。
"""
import os
import pandas as pd


if __name__ == '__main__':
    root_path = 'D:\datasets\dataset\ETT-small'
    data_path = 'ETTh1.csv'
    df_raw = pd.read_csv(os.path.join(root_path, data_path))
    print(df_raw.head())
    print(df_raw.info)
