#!/usr/bin/env bash
set -e   # 任一步报错就退出

for emo in surprise joy sadness; do
  echo "=== training $emo ==="
  python train.py \
    --output_dir "outputs/$emo" \
    --backbone roberta-base \
    --prompt_len 100 \
    --sentiment "$emo" \
    --max_source_length 128 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 5e-5 \
    --num_train_epochs 10 \
    --seed 42
done

# 终端运行：
# chmod +x run.bash 赋执行权限（只需一次）
# ./run.bash