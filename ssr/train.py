import os
import torch
import autoroot
import transformers
from torch import nn
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
from ssr.utils.load_model import load_smamba, load_internlm3


@dataclass
class ModelArguments:
    mamba: str = field(default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "mamba-130m-hf"))
    llm: str = field(default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "internlm3-8b-instruct"))


@dataclass
class DataArguments:
    pass


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    pass


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    rank0_print(training_args.local_rank, f"{str_datetime()} Loading Mamba ...")
    smamba = load_smamba(model_args.mamba)
    rank0_print(training_args.local_rank, f"{str_datetime()} Loading LLM ...")
    ssr = load_internlm3(model_args.llm, bits=4)


if __name__ == "__main__":
    train()