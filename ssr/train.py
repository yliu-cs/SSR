import os
import wandb
import torch
import autoroot
import transformers
from glob import glob
from typing import Tuple
from ssr.utils.misc import (
    init
    , rank0_print
    , count_params
    , str_datetime
)
from transformers import Trainer
from dataclasses import dataclass, field
from ssr.models.modeling_ssr import SSR, SSRConfig
from ssr.utils.load_ptm import load_clip_vit, load_siglip
from ssr.data.data import prepare_ssr_dataset, SSRDataCollator


@dataclass
class ModelArguments:
    mamba_path: str = field(default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "mamba-130m-hf"))
    internlm3_path: str = field(default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "internlm3-8b-instruct"))
    clip_path: str = field(default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "clip-vit-large-patch14-336"))
    siglip_path: str = field(default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "siglip-so400m-patch14-384"))
    depth_pro_path: str = field(default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "DepthPro"))
    stage1_path: str = field(default=os.path.join(os.getcwd(), "checkpoint", "SSR", "stage1"))


@dataclass
class DataArguments:
    cot_data_dirs: list[str] = field(default_factory=lambda: [
        os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "LLaVA-CoT-100k")
        , os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "Visual-CoT")
        , os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "VoCoT")
        , os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "SpatialQA")
    ])
    n_tor: int = field(default=10)
    n_image_tokens: int = field(default=(336 // 14) ** 2)
    n_depth_tokens: int = field(default=(384 // 14) ** 2)
    max_length: Tuple[int, int, int] = field(default=(256, 1024, 256))


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    bits: int = field(default=4)
    stage: int = field(default=1)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=64)
    lora_dropout: float = field(default=0.1)


def train() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    wandb.login(key="597d3699174e2903f391fa8918967d5ada6fc99d")

    rank0_print(training_args.local_rank, f"{str_datetime()} Loading CLIPVisionModel ...")
    clip_processor, clip_vision = load_clip_vit(model_args.clip_path)
    rank0_print(training_args.local_rank, f"{str_datetime()} Loading SigLIP ...")
    siglip_processor, siglip = load_siglip(model_args.siglip_path)

    rank0_print(training_args.local_rank, f"{str_datetime()} Loading SSR ...")
    ssr_config = SSRConfig(
        mamba_path=model_args.mamba_path
        , internlm3_path=model_args.internlm3_path
        , stage=training_args.stage
        , lora_r=training_args.lora_r
        , lora_alpha=training_args.lora_alpha
        , lora_dropout=training_args.lora_dropout
    )
    if training_args.stage != 1:
        ssr_config.stage = training_args.stage
        setattr(ssr_config, f"stage{training_args.stage - 1}_path", getattr(model_args, f"stage{training_args.stage - 1}_path"))
    ssr = SSR(
        config=ssr_config
        , clip_vision=clip_vision
        , siglip=siglip
    )
    ssr.prepare_modules()
    rank0_print(training_args.local_rank, f"{str_datetime()} {'SSR':<30} {count_params(ssr)}")
    rank0_print(training_args.local_rank, f"{str_datetime()} {'Image Encoder':<30} {count_params(ssr.image_encoder)}")
    rank0_print(training_args.local_rank, f"{str_datetime()} {'Depth Encoder':<30} {count_params(ssr.depth_encoder)}")
    rank0_print(training_args.local_rank, f"{str_datetime()} {'Mamba Image Projector':<30} {count_params(ssr.mamba_image_proj)}")
    rank0_print(training_args.local_rank, f"{str_datetime()} {'Mamba Depth Projector':<30} {count_params(ssr.mamba_depth_proj)}")
    rank0_print(training_args.local_rank, f"{str_datetime()} {'Mamba':<30} {count_params(ssr.mamba)}")
    rank0_print(training_args.local_rank, f"{str_datetime()} {'Tor Projector':<30} {count_params(ssr.tor_proj)}")
    rank0_print(training_args.local_rank, f"{str_datetime()} {'InternLM3 Image Projector':<30} {count_params(ssr.internlm3_image_proj)}")
    rank0_print(training_args.local_rank, f"{str_datetime()} {'InternLM3 Depth Projector':<30} {count_params(ssr.internlm3_depth_proj)}")
    rank0_print(training_args.local_rank, f"{str_datetime()} {'InternLM3':<30} {count_params(ssr.internlm3)}")

    dataset = prepare_ssr_dataset(
        data_args.cot_data_dirs
        , tokenizer=ssr.tokenizer
        , max_length=data_args.max_length
        , clip_processor=clip_processor
        , siglip_processor=siglip_processor
    )
    rank0_print(training_args.local_rank, f"{str_datetime()} Dataset Size: {len(dataset)}")
    trainer = Trainer(
        model=ssr
        , args=training_args
        , train_dataset=dataset
        , data_collator=SSRDataCollator(
            stage=training_args.stage
            , n_tor=data_args.n_tor
            , n_image_tokens=data_args.n_image_tokens
            , n_depth_tokens=data_args.n_depth_tokens
            , tokenizer=ssr.tokenizer
        )
    )
    trainer.train(resume_from_checkpoint=(True if list(glob(os.path.join(training_args.output_dir, "checkpoint-*"))) else False))
    trainer.save_state()
    torch.cuda.synchronize()
    if training_args.local_rank == 0:
        rank0_print(training_args.local_rank, f"{str_datetime()} Saving SSR ...")
        unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
        unwrapped_model.config.save_pretrained(training_args.output_dir)
        unwrapped_model.tokenizer.save_pretrained(os.path.join(training_args.output_dir, "internlm3"))
        for module_name in ["mamba", "internlm3", "mamba_image_proj", "mamba_depth_proj", "tor_proj", "internlm3_image_proj", "internlm3_depth_proj"]:
            if module_name == "internlm3" and training_args.stage == 1:
                continue
            module = getattr(unwrapped_model, module_name)
            module_path = os.path.join(training_args.output_dir, module_name)
            module.save_pretrained(
                module_path
                , state_dict=trainer.accelerator.get_state_dict(module)
                , safe_serialization=trainer.args.save_safetensors
            )
        rank0_print(training_args.local_rank, f"{str_datetime()} Saving SSR Done.")

if __name__ == "__main__":
    init()
    train()