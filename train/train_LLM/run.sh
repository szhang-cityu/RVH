#!/bin/bash
#SBATCH --job-name=alphaedit         # 作业名称-请修改为自己任务名字 
#SBATCH --cpus-per-task=4             # 每个任务使用的CPU核心数
#SBATCH --output=/home/szhang844/SFT/log.txt        # 标准输出文件名 (%j 表示作业ID)-请修改为自己路径
#SBATCH --error=/home/szhang844/SFT/error.txt 
#SBATCH --mem=200G                      # 申请100GB内存
#SBATCH --time=24:00:00               # 运行时间限制，格式为hh:mm:ss
python lora_prepare_and_train_qwen3.py