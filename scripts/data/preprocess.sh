#!/bin/bash

NGPUS=$(python -c "import torch; print(torch.cuda.device_count())")
for i in $(seq 0 $((NGPUS - 1))); do
    echo "Starting data preprocess $i on GPU $i ($NGPUS GPUs)"
    CUDA_VISIBLE_DEVICES=$i python ssr/data/preprocess.py --num_chunks $NGPUS --chunk_idx $i &
done
wait
echo "All preprocess tasks finished."