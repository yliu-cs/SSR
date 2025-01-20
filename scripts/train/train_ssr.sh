NGPUS=$(python -c "import torch; print(torch.cuda.device_count())")
WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23456}

torchrun \
    --nnodes=${WORLD_SIZE} \
    --node_rank=${RANK} \
    --nproc_per_node=${NGPUS} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    ssr/train.py \
    --deepspeed scripts/ds_zero2.json \
    --output_dir ./checkpoint/SSR \
    --bf16 True \
    --per_device_train_batch_size 8 \
    --eval_strategy no \
    --save_strategy epoch \
    --save_only_model True \
    --save_total_limit 1 \
    --num_train_epochs 2 \
    --learning_rate 3e-5 \
    --lr_scheduler_type cosine \
    --warmup_steps 800 \
    --logging_steps 1 \
    --report_to none