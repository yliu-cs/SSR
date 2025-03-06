#!/bin/bash

NGPUS=$(python -c "import torch; print(torch.cuda.device_count())")
for i in $(seq 0 $((NGPUS - 1))); do
    echo "Starting task $i on GPU $i ($NGPUS GPUs)"
    CUDA_VISIBLE_DEVICES=$i python ssr/eval/explicit_cot.py --num_chunks $NGPUS --chunk_idx $i &
done
wait

python -c 'import os
import numpy as np
from glob import glob

qa_preds, qa_scores, qra_preds, qra_scores = [], [], [], []
for file in glob(os.path.join(os.getcwd(), "media", "exp_cot_eval", "qa_preds_*.npy")):
    qa_preds += np.load(file).tolist()
    os.remove(file)
for file in glob(os.path.join(os.getcwd(), "media", "exp_cot_eval", "qa_scores_*.npy")):
    qa_scores += np.load(file).tolist()
    os.remove(file)
for file in glob(os.path.join(os.getcwd(), "media", "exp_cot_eval", "qra_preds_*.npy")):
    qra_preds += np.load(file).tolist()
    os.remove(file)
for file in glob(os.path.join(os.getcwd(), "media", "exp_cot_eval", "qra_scores_*.npy")):
    qra_scores += np.load(file).tolist()
    os.remove(file)
np.save(os.path.join(os.getcwd(), "media", "exp_cot_eval", "qa_preds.npy"), qa_preds)
np.save(os.path.join(os.getcwd(), "media", "exp_cot_eval", "qa_scores.npy"), qa_scores)
np.save(os.path.join(os.getcwd(), "media", "exp_cot_eval", "qra_preds.npy"), qra_preds)
np.save(os.path.join(os.getcwd(), "media", "exp_cot_eval", "qra_scores.npy"), qra_scores)

print(f"QA Accuracy: {sum(qa_preds) / len(qa_preds):.4f} QA Score: {np.mean(qa_scores):.4f}")
print(f"QRA Accuracy: {sum(qra_preds) / len(qra_preds):.4f} QRA Score: {np.mean(qra_scores):.4f}")'