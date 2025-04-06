#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2025/4/6 14:55
 @Author  : wly
 @File    : 时间序列autocorrelation.py
 @Description: 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # making Time series
    spacing = np.linspace(-5 * np.pi, 5 * np.pi, num=100)
    s = pd.Series(0.7 * np.random.rand(100) + 0.3 * np.sin(spacing))

    # Plotting the Time series
    plt.plot(spacing, s)
    plt.show()

    # Creating Autocorrelation plot
    x = pd.plotting.autocorrelation_plot(s)
    # plotting the Curve
    x.plot()
    # Display
    plt.show()
