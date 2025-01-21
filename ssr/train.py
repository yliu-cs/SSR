import os
import torch
import autoroot
import transformers
from glob import glob
from ssr.utils.misc import (
    init
    , rank0_print
    , count_params
    , str_datetime
)
from transformers import Trainer
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from ssr.models.modeling_ssr import SSR, SSRConfig
from ssr.data.data import prepare_ssr_dataset, SSRDataCollator
from ssr.utils.load_ptm import load_clip_vit, load_siglip, load_depth_pro


@dataclass
class ModelArguments:
    mamba_path: str = field(default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "mamba-130m-hf"))
    internlm3_path: str = field(default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "internlm3-8b-instruct"))
    clip_path: str = field(default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "clip-vit-large-patch14-336"))
    siglip_path: str = field(default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "siglip-so400m-patch14-384"))
    depth_pro_path: str = field(default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "DepthPro"))
    n_tor: int = field(default=10)


@dataclass
class DataArguments:
    cot_data_names: list[str] = field(default_factory=lambda: [
        os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "VRC-Bench")
    ])


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    bits: int = field(default=16)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    clip_processor, clip_vision = load_clip_vit(model_args.clip_path)
    siglip_processor, siglip = load_siglip(model_args.siglip_path)
    depth_pro, depth_transform = load_depth_pro(model_args.depth_pro_path)
    depth_pro = depth_pro.to(training_args.device)

    rank0_print(training_args.local_rank, f"{str_datetime()} Loading SSR ...")
    ssr = SSR(
        SSRConfig(
            mamba_path=model_args.mamba_path
            , internlm3_path=model_args.internlm3_path
            , bits=training_args.bits
        )
        , clip_vision=clip_vision
        , siglip=siglip
    )
    ssr = ssr.to_empty(device=training_args.device)
    rank0_print(training_args.local_rank, f"{str_datetime()} {count_params(ssr)}")
    rank0_print(training_args.local_rank, f"{str_datetime()} {count_params(ssr.mamba)}")
    rank0_print(training_args.local_rank, f"{str_datetime()} {count_params(ssr.internlm3)}")

    trainer = Trainer(
        model=ssr
        , args=training_args
        , train_dataset=prepare_ssr_dataset(
            data_args.cot_data_names
            , clip_processor=clip_processor
            , siglip_processor=siglip_processor
            , depth_pro=depth_pro
            , depth_transform=depth_transform
        )
        , data_collator=SSRDataCollator()
    )
    trainer.train(resume_from_checkpoint=(True if list(glob(os.path.join(training_args.output_dir, "checkpoint-*"))) else False))
    trainer.save_state()
    torch.cuda.synchronize()
    if training_args.local_rank == 0:
        ssr.config.save_pretrained(training_args.output_dir)
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    train()