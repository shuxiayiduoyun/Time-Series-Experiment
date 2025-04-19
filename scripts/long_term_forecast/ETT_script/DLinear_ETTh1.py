#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2025/4/9 22:52
 @Author  : wly
 @File    : DLinear_ETTh1.py
 @Description: 
"""
import subprocess
import os
# 切换到run.py所在的目录
os.chdir("E:\workspaces\workspace_python\Time-Series-Experiment")
model_name = "DLinear"
root_pth = "E:\workspaces\workspace_python\Time-Series-Experiment\dataset\ETT-small"
commands = [
    f"python -u run.py --task_name long_term_forecast --is_training 1 --root_path {root_pth} --data_path ETTh1.csv --model_id ETTh1_96_96 --model {model_name} --data ETTh1 --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7",
    # f"python -u run.py --task_name long_term_forecast --is_training 1 --root_path {root_pth} --data_path ETTh1.csv --model_id ETTh1_96_192 --model {model_name} --data ETTh1 --features M --seq_len 96 --label_len 48 --pred_len 192 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7",
    # f"python -u run.py --task_name long_term_forecast --is_training 1 --root_path {root_pth} --data_path ETTh1.csv --model_id ETTh1_96_336 --model {model_name} --data ETTh1 --features M --seq_len 96 --label_len 48 --pred_len 336 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7",
    # f"python -u run.py --task_name long_term_forecast --is_training 1 --root_path {root_pth} --data_path ETTh1.csv --model_id ETTh1_96_720 --model {model_name} --data ETTh1 --features M --seq_len 96 --label_len 48 --pred_len 720 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7"
]

for cmd in commands:
    subprocess.run(cmd, shell=True, check=True)
