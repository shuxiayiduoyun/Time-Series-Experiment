#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2025/4/6 15:09
 @Author  : wly
 @File    : 自相关系数计算.py
 @Description: 
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf


def compute_acf(series, max_lag):
    n = len(series)
    mean = np.mean(series)
    centered = series - mean
    denominator = np.sum(centered ** 2)
    acf = [1.0]  # r_0 = 1

    for k in range(1, max_lag + 1):
        if k < n:
            numerator = np.sum(centered[:n - k] * centered[k:])
            acf_k = numerator / denominator
        else:
            acf_k = 0  # 滞后超出范围时返回0
        acf.append(acf_k)
    return acf


if __name__ == '__main__':
    data = [1, 2, 3, 4, 5]
    max_lag = 2
    print(compute_acf(data, max_lag))  # 输出: [1.0, 0.4, -0.1]
    acf_values = acf(data, nlags=2, adjusted=False, fft=False)
    print(acf_values)
    lag_1 = pd.Series(np.array(data)).autocorr(lag=1)
    print(lag_1)
