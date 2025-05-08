#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2025/5/8 10:34
 @Author  : wly
 @File    : DLinear_script.py
 @Description: 
"""
import subprocess
import os


os.chdir("F:\workspaces\workspace_python\Time-Series-Experiment")
model_name = "DLinear"
commands = [
    # f"python -u run.py --task_name classification --is_training 1 --root_path ./dataset/EthanolConcentration/ --model_id EthanolConcentration --model {model_name} --data UEA --e_layers 3 --batch_size 16 --d_model 128 --d_ff 256 --top_k 3 --des 'Exp' --itr 1 --learning_rate 0.001 --train_epochs 100 --patience 10",
    # f"python -u run.py --task_name classification --is_training 1 --root_path ./dataset/FaceDetection/ --model_id FaceDetection --model {model_name} --data UEA --e_layers 3 --batch_size 16 --d_model 128 --d_ff 256 --top_k 3 --des 'Exp' --itr 1 --learning_rate 0.001 --train_epochs 100 --patience 10",
    # f"python -u run.py --task_name classification --is_training 1 --root_path ./dataset/Handwriting/ --model_id Handwriting --model {model_name} --data UEA --e_layers 3 --batch_size 16 --d_model 128 --d_ff 256 --top_k 3 --des 'Exp' --itr 1 --learning_rate 0.001 --train_epochs 100 --patience 10",
    # f"python -u run.py --task_name classification --is_training 1 --root_path ./dataset/Heartbeat/ --model_id Heartbeat --model {model_name} --data UEA --e_layers 3 --batch_size 16 --d_model 128 --d_ff 256 --top_k 3 --des 'Exp' --itr 1 --learning_rate 0.001 --train_epochs 100 --patience 10",
    # f"python -u run.py --task_name classification --is_training 1 --root_path ./dataset/JapaneseVowels/ --model_id JapaneseVowels --model {model_name} --data UEA --e_layers 3 --batch_size 16 --d_model 128 --d_ff 256 --top_k 3 --des 'Exp' --itr 1 --learning_rate 0.001 --train_epochs 100 --patience 10",
    # f"python -u run.py --task_name classification --is_training 1 --root_path ./dataset/PEMS-SF/ --model_id PEMS-SF --model {model_name} --data UEA --e_layers 3 --batch_size 16 --d_model 128 --d_ff 256 --top_k 3 --des 'Exp' --itr 1 --learning_rate 0.001 --train_epochs 100 --patience 10",
    # f"python -u run.py --task_name classification --is_training 1 --root_path ./dataset/SelfRegulationSCP1/ --model_id SelfRegulationSCP1 --model {model_name} --data UEA --e_layers 3 --batch_size 16 --d_model 128 --d_ff 256 --top_k 3 --des 'Exp' --itr 1 --learning_rate 0.001 --train_epochs 100 --patience 10",
    # f"python -u run.py --task_name classification --is_training 1 --root_path ./dataset/SelfRegulationSCP2/ --model_id SelfRegulationSCP2 --model {model_name} --data UEA --e_layers 3 --batch_size 16 --d_model 128 --d_ff 256 --top_k 3 --des 'Exp' --itr 1 --learning_rate 0.001 --train_epochs 100 --patience 10",
    f"python -u run.py --task_name classification --is_training 1 --root_path D:\datasets\dataset\SpokenArabicDigits\ --model_id SpokenArabicDigits --model {model_name} --data UEA --e_layers 3 --batch_size 16 --d_model 128 --d_ff 256 --top_k 3 --learning_rate 0.001 --train_epochs 100 --patience 10",
    # f"python -u run.py --task_name classification --is_training 1 --root_path ./dataset/UWaveGestureLibrary/ --model_id UWaveGestureLibrary --model {model_name} --data UEA --e_layers 3 --batch_size 16 --d_model 128 --d_ff 256 --top_k 3 --des 'Exp' --itr 1 --learning_rate 0.001 --train_epochs 100 --patience 10"
]

for cmd in commands:
    subprocess.run(cmd, shell=True, check=True)
