#!/bin/bash

python /home/szhang844/SFT/grpo_train_qwen3.py \
  --rl_json /home/szhang844/SFT/abnormal_rl.json \
  --base_model /home/szhang844/SFT/merge2 \
  --output_dir /home/szhang844/SFT/grpo_lora2 \
  --metrics_csv /home/szhang844/SFT/results/grpo_metrics2.csv \
  --max_steps 200 \
  --num_generations 4 \
  --max_new_tokens 256
