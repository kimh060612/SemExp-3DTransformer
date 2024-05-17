#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 train.py --split val --eval 1 --load pretrained_models/periodic_8000000.pth -n 1 --auto_gpu_config=0
CUDA_VISIBLE_DEVICES=0 python3 train.py -n 1 --auto_gpu_config=0 --num_global_steps=5 --num_mini_batch=3
CUDA_VISIBLE_DEVICES=0 python3 test.py -n 1 --auto_gpu_config=0 --num_global_steps=5 --num_mini_batch=3

