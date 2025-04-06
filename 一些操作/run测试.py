#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2025/4/6 20:13
 @Author  : wly
 @File    : run测试.py
 @Description: 
"""
import random
import torch
import numpy as np
import argparse

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


if __name__ == '__main__':
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='run测试')
    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='ETTh1_96_96')
    parser.add_argument('--model', type=str, default='Transformer')
    # data loader
    parser.add_argument('--data', type=str, default='ETTh1')
    parser.add_argument('--root_path', type=str, default='D:\datasets\dataset\ETT-small')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True)
    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    print('Args in experiment:')
    print(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
