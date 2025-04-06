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
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt


if __name__ == '__main__':
    root_path = 'D:\datasets\dataset\ETT-small'
    data_path = 'ETTh1.csv'
    df_raw = pd.read_csv(os.path.join(root_path, data_path))
    print(df_raw.head())
    print(df_raw.info)
    print(df_raw.columns)

    if 1 == 1:
        # 绘制OT数据
        date_axis = pd.to_datetime(df_raw['date'])
        plt.figure(figsize=(12, 6))
        plt.plot(date_axis, df_raw['OT'], label='OT')
        plt.xlabel('Date')
        plt.ylabel('OT')
        plt.title('OT over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

    if 1 == 1:
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        # 选择连续的 500 个点
        start_index = 0  # 你可以根据需要调整起始索引
        df_subset = df_raw.iloc[start_index:start_index + 500]
        plt.figure(figsize=(12, 6))
        for col in df_subset.columns:
            if col != 'date':  # 排除 'date' 列，因为它是横轴
                plt.plot(df_subset['date'], df_subset[col], label=col)

        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('All Features over Time (500 Points)')
        plt.legend()
        plt.grid(True)
        plt.show()

    df_raw['date'] = pd.to_datetime(df_raw['date'])
    start_index = 0  # 你可以根据需要调整起始索引
    df_subset = df_raw.iloc[start_index:start_index + 500]
    plt.figure(figsize=(12, 8))
    for col in df_subset.columns:
        if col != 'date':  # 排除 'date' 列，因为它是时间戳
            autocorrelation_plot(df_subset[col], label=col)

    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation Plot of All Variables in ETTh1 Dataset')
    plt.legend()
    plt.grid(True)
    plt.show()
