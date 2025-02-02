#!/bin/bash

for i in {0..7}; do
    echo "Starting task on GPU $i"
    # 使用 CUDA_VISIBLE_DEVICES 指定显卡索引
    CUDA_VISIBLE_DEVICES=$i python ssr/data/llava_cot.py --chunk_idx $i &
done
wait
echo "All tasks finished."