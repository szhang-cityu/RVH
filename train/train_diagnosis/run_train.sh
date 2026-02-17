#!/usr/bin/env bash
set -e

conda activate rvh
python train_diagnosis_model.py --abnormal abnormal_2000.dat --normal normal_2000.dat --save model-mamba-cnn.pt
