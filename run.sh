#!/usr/bin/env bash
rm out.MD_LSTM.txt
rm out.SI_LSTM.txt
export CUDA_VISIBLE_DEVICES=; nohup python3 -u main.py 0 > out.SI_LSTM.txt 2>&1 &
export CUDA_VISIBLE_DEVICES=; nohup python3 -u main.py 1 > out.MD_LSTM.txt 2>&1 &
