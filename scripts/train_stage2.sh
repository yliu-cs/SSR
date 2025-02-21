if [ -f "scripts/stage2_flag.txt" ]; then
    echo "stage2_flag.txt exist, exit"
    exit 0
else
    touch "scripts/stage2_flag.txt"
    echo "create stage2_flag.txt, continue"
fi

NGPUS=$(python -c "import torch; print(torch.cuda.device_count())")
RANK=${RANK:-0}
WORLD_SIZE=${WORLD_SIZE:-1}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23456}

torchrun \
    --nnodes=${WORLD_SIZE} \
    --node_rank=${RANK} \
    --nproc_per_node=${NGPUS} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    ssr/train.py \
    --output_dir /ssdwork/liuyang/Checkpoint/SSR/stage2 \
    --stage1_path /ssdwork/liuyang/Checkpoint/SSR/stage1 \
    --per_device_train_batch_size 8 \
    --remove_unused_columns False \
    --ddp_find_unused_parameters True \
    --stage 2 \
    --bf16 True \
    --eval_strategy no \
    --save_strategy epoch \
    --save_only_model True \
    --save_total_limit 1 \
    --save_steps 0.2 \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.01 \
    --logging_steps 1 \
    --dataloader_num_workers 30 \
    --dataloader_pin_memory True \
    --dataloader_persistent_workers True \
    --dataloader_prefetch_factor 2 \
    --fsdp_config=./scripts/fsdp.json \
    --report_to wandb \
    --run_name SSR/Stage2