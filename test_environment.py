# encoding: utf-8
"""
@author: wly
@software: PyCharm
@file: test_environment.py
@time: 2025/4/5 20:21
"""
import torch
import sys

# 检查CUDA是否可用
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("device_name: {}".format(torch.cuda.get_device_name(0)))
# python版本
print("python version: {}".format(sys.version))
