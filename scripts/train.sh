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
    --output_dir ./checkpoint/SSR/stage1 \
    --per_device_train_batch_size 4 \
    --remove_unused_columns False \
    --stage SSRStage.mamba \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 1000 \
    --save_only_model True \
    --max_steps 50000 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_steps 400 \
    --logging_steps 1 \
    --report_to none


torchrun \
    --nnodes=${WORLD_SIZE} \
    --node_rank=${RANK} \
    --nproc_per_node=${NGPUS} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    ssr/train.py \
    --output_dir ./checkpoint/SSR/stage2 \
    --per_device_train_batch_size 2 \
    --remove_unused_columns False \
    --stage SSRStage.internlm \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 1000 \
    --save_only_model True \
    --max_steps 50000 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_steps 400 \
    --logging_steps 1 \
    --report_to none