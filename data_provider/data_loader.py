#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2025/4/6 16:01
 @Author  : wly
 @File    : data_loader.py
 @Description: 
"""
import os
# import numpy as np
import pandas as pd
# import glob
# import re
# import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
# from sktime.datasets import load_from_tsfile_to_dataframe
# import warnings

# warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv', target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # 根据数据集的类型（训练集、验证集或测试集），计算数据的分割边界 border1 和 border2，用于从完整数据集中提取对应的子集
        # border1s：每个数据集的起始索引列表
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        # border2s：每个数据集的结束索引列表
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        """
        border1s：
        0：训练集的起始索引。
        12 * 30 * 24 - self.seq_len：验证集的起始索引。
        12 * 30 * 24 + 4 * 30 * 24 - self.seq_len：测试集的起始索引。
        border2s：
        12 * 30 * 24：训练集的结束索引。
        12 * 30 * 24 + 4 * 30 * 24：验证集的结束索引。
        12 * 30 * 24 + 8 * 30 * 24：测试集的结束索引。
        """
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            # 'M' 或 'MS'：选择多变量（Multivariate）特征，即除了时间戳列之外的所有列
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            # 'S'：选择单变量（Univariate）特征，即仅选择目标变量列
            df_data = df_raw[[self.target]]

        if self.scale:
            # 仅使用训练集计算scale
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(labels=['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        # 输入序列的起始索引，直接使用传入的index
        s_begin = index
        # 输入序列的结束索引
        s_end = s_begin + self.seq_len
        # 目标序列的起始索引
        r_begin = s_end - self.label_len
        # 目标序列的结束索引
        r_end = r_begin + self.label_len + self.pred_len
        # 输入序列
        seq_x = self.data_x[s_begin:s_end]
        # 目标序列
        seq_y = self.data_y[r_begin:r_end]
        # 输入序列的时间戳特征
        seq_x_mark = self.data_stamp[s_begin:s_end]
        # 目标序列的时间戳特征
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
