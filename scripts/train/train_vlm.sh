if [ -f "scripts/train_vlm.flag" ]; then
    echo "train_vlm.flag exist, exit"
    exit 0
else
    touch "scripts/train_vlm.flag"
    echo "create train_vlm.flag, continue"
fi

accelerate launch --config_file "scripts/fsdp.yaml" ssr/train/train_vlm.py --lora --llava