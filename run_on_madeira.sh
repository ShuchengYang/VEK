#!/bin/bash
#SBATCH --job-name=syang-vek-rwqa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --time=15:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shucheng.yang@tum.de
#SBATCH --output=yangshucheng-compass/slurm-logs/rwqa_madeira_log_%j.out

source ~/.bashrc
conda activate evalkit

# config zone
DS="RealWorldQA"
# end of config zone

DATE="$(date +%Y%m%d_%H%M%S)"
WORK_DIR="yangshucheng-compass/${DS}_result_${DATE}"
echo "result in ./$WORK_DIR"


PYTHONUNBUFFERED=1 torchrun run.py \
  --nproc-per-node=2 \
  --master_port=12345 \
 --data "$DS" \
 --model Qwen2.5-VL-7B-Tool-Code \
 --verbose \
 --work-dir "$WORK_DIR"