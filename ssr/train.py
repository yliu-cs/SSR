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
from ssr.utils.prompt import SSRStage
from dataclasses import dataclass, field
from ssr.models.modeling_ssr import SSR, SSRConfig
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from ssr.data.data import prepare_ssr_dataset, SSRDataCollator
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from ssr.utils.load_ptm import load_clip_vit, load_siglip, load_depth_pro


@dataclass
class ModelArguments:
    mamba_path: str = field(default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "mamba-130m-hf"))
    internlm3_path: str = field(default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "internlm3-8b-instruct"))
    clip_path: str = field(default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "clip-vit-large-patch14-336"))
    siglip_path: str = field(default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "siglip-so400m-patch14-384"))
    depth_pro_path: str = field(default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "DepthPro"))


@dataclass
class DataArguments:
    cot_data_names: list[str] = field(default_factory=lambda: [
        os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "VRC-Bench")
    ])
    n_tor: int = field(default=10)
    n_image_tokens: int = field(default=(336 // 14) ** 2)
    n_depth_tokens: int = field(default=(384 // 14) ** 2)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    bits: int = field(default=4)
    stage: str = field(default="SSRStage.mamba")
    bf16: bool = field(default=False)
    fp16: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=64)
    lora_dropout: float = field(default=0.1)
    fsdp_proj_mp: dict = field(
        default_factory=lambda: {
            "param_dtype": "float32",
            "reduce_dtype": "float32",
            "buffer_dtype": "float32",
        }
    )
    fsdp_mamba_mp: dict = field(
        default_factory=lambda: {
            "param_dtype": "float32",
            "reduce_dtype": "float32",
            "buffer_dtype": "float32",
        }
    )
    fsdp_internlm3_mp: dict = field(
        default_factory=lambda: {
            "param_dtype": "bfloat16",
            "reduce_dtype": "bfloat16",
            "buffer_dtype": "bfloat16",
        }
    )


def train() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.stage = eval(training_args.stage)

    rank0_print(training_args.local_rank, f"{str_datetime()} Loading CLIPVisionModel ...")
    clip_processor, clip_vision = load_clip_vit(model_args.clip_path, device=training_args.device)
    rank0_print(training_args.local_rank, f"{str_datetime()} Loading SigLIP ...")
    siglip_processor, siglip = load_siglip(model_args.siglip_path, device=training_args.device)
    rank0_print(training_args.local_rank, f"{str_datetime()} Loading DepthPro ...")
    depth_pro, depth_transform = load_depth_pro(model_args.depth_pro_path, device=training_args.device)

    rank0_print(training_args.local_rank, f"{str_datetime()} Loading SSR ...")
    ssr = SSR(
        SSRConfig(
            mamba_path=model_args.mamba_path
            , internlm3_path=model_args.internlm3_path
            , bits=training_args.bits
            , stage=training_args.stage
            , lora_r=training_args.lora_r
            , lora_alpha=training_args.lora_alpha
            , lora_dropout=training_args.lora_dropout
            , device=training_args.device
        )
        , clip_vision=clip_vision
        , siglip=siglip
    )
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

    mamba_mp = MixedPrecision(
        param_dtype=getattr(torch, training_args.fsdp_mamba_mp["param_dtype"]),
        reduce_dtype=getattr(torch, training_args.fsdp_mamba_mp["reduce_dtype"]),
        buffer_dtype=getattr(torch, training_args.fsdp_mamba_mp["buffer_dtype"]),
    )
    proj_mp = MixedPrecision(
        param_dtype=getattr(torch, training_args.fsdp_proj_mp["param_dtype"]),
        reduce_dtype=getattr(torch, training_args.fsdp_proj_mp["reduce_dtype"]),
        buffer_dtype=getattr(torch, training_args.fsdp_proj_mp["buffer_dtype"]),
    )
    internlm3_mp = MixedPrecision(
        param_dtype=getattr(torch, training_args.fsdp_internlm3_mp["param_dtype"]),
        reduce_dtype=getattr(torch, training_args.fsdp_internlm3_mp["reduce_dtype"]),
        buffer_dtype=getattr(torch, training_args.fsdp_internlm3_mp["buffer_dtype"]),
    )
    training_args.fsdp_config = {
        "xla": False
        , "fsdp_auto_wrap_policy": {
            "policy": ModuleWrapPolicy
            , "module_classes": {
                ssr.mamba
                , ssr.mamba_image_proj
                , ssr.mamba_depth_proj
                , ssr.internlm3
                , ssr.tor_proj
                , ssr.internlm3_image_proj
                , ssr.internlm3_depth_proj
            }
        }
        , "mixed_precision_policies": {
            ssr.mamba: mamba_mp
            , ssr.mamba_image_proj: proj_mp
            , ssr.mamba_depth_proj: proj_mp
            , ssr.internlm3: internlm3_mp
            , ssr.tor_proj: proj_mp
            , ssr.internlm3_image_proj: proj_mp
            , ssr.internlm3_depth_proj: proj_mp
        }
        , "min_num_params": 1e4
        , "sharding_strategy": ShardingStrategy.SHARD_GRAD_OP
        , "shard_init": True
        , "offload_params": False
        , "activation_checkpointing": True
    }

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
        ssr.config.save_pretrained(training_args.output_dir)
        ssr.tokenizer.save_pretrained(training_args.output_dir)
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    init()
    train()