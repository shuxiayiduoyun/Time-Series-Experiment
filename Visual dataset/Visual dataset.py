# encoding: utf-8
"""
@author: wly
@software: PyCharm
@file: Visual dataset.py
@time: 2025/4/6 10:43
"""
import os
import pandas as pd


if __name__ == '__main__':
    root_path = '../dataset/ETT-small'
    data_path = 'ETTh1.csv'
    df_raw = pd.read_csv(os.path.join(root_path, data_path))
