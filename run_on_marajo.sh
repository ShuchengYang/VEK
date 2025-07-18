#!/usr/bin/env bash
set -euo pipefail

source ~/.bashrc
conda activate evalkit
export TRANSFORMERS_VERBOSITY=error

# ─── 配置区 ───────────────────────────────────
DS="${DS:-MMStar}"
CODE="${CODE:-true}"
# ──────────────────────────────────────────────

DATE="$(date +%Y%m%d_%H%M%S)"
WORK_DIR="yangshucheng-compass/${DS}_result_${DATE}"
echo "result in ./$WORK_DIR"

# 根据 CODE 的值选择执行哪条命令
if [[ "$CODE" == "true" ]]; then
  python run.py \
    --data "$DS" \
    --model Qwen2.5-VL-7B-Tool-Code \
    --verbose \
    --work-dir "$WORK_DIR"
else
  python run.py \
    --data "$DS" \
    --model Qwen2.5-VL-7B-Answer \
    --verbose \
    --work-dir "$WORK_DIR"
fi

