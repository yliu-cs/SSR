if [ -f "scripts/train_reasoning.flag" ]; then
    echo "train_reasoning.flag exist, exit"
    exit 0
else
    touch "scripts/train_reasoning.flag"
    echo "create train_reasoning.flag, continue"
fi

accelerate launch --config_file "scripts/fsdp.yaml" ssr/train/train_reasoning.py