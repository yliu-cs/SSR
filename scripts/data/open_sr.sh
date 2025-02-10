#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <start_index>"
    exit 1
fi
start_index=$1
for i in $(seq $start_index $((start_index + 7))); do
    echo "Starting task $i on GPU $((i-start_index))"
    CUDA_VISIBLE_DEVICES=$((i-start_index)) python ssr/data/open_sr.py --chunk_idx $i &
done
wait
echo "All tasks finished."