#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2025/4/6 18:18
 @Author  : wly
 @File    : dataset测试.py
 @Description: 
"""
from data_provider.data_loader import Dataset_ETT_hour

if __name__ == '__main__':
    root_path = 'D:\datasets\dataset\ETT-small'
    data_path = 'ETTh1.csv'
    seq_len, label_len, pred_len = 96, 48, 96
    size = [seq_len, label_len, pred_len]
    dataset = Dataset_ETT_hour(
        root_path=root_path,
        data_path=data_path,
        flag='train',
        features='M',
        size=size,
        target='OT',
        timeenc=0,
        scale=True,
        freq='h',
        seasonal_patterns='Monthly'
    )
    print(f'length of dataset: {len(dataset)}.')
    print(dataset.__getitem__(index=8544)[0].shape)
    print(dataset.__getitem__(index=8544)[1].shape)
    print(type(dataset.__getitem__(index=8544)), len(dataset.__getitem__(index=8544)))
