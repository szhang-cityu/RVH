#!/bin/bash
#SBATCH --job-name=alphaedit         # 作业名称-请修改为自己任务名字 
#SBATCH --cpus-per-task=4             # 每个任务使用的CPU核心数
#SBATCH --output=/home/szhang844/SFT/RL-log.txt        # 标准输出文件名 (%j 表示作业ID)-请修改为自己路径
#SBATCH --error=/home/szhang844/SFT/RL-error.txt 
#SBATCH --mem=200G                      # 申请100GB内存
#SBATCH --time=24:00:00               # 运行时间限制，格式为hh:mm:ss

python /home/szhang844/SFT/grpo_train_qwen3.py \
  --rl_json /home/szhang844/SFT/abnormal_rl.json \
  --base_model /home/szhang844/SFT/merge2 \
  --output_dir /home/szhang844/SFT/grpo_lora2 \
  --metrics_csv /home/szhang844/SFT/results/grpo_metrics2.csv \
  --max_steps 200 \
  --num_generations 4 \
  --max_new_tokens 256