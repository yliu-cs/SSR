import os
import pytz
import math
import torch
import warnings
import transformers
from torch import nn
from typing import List, Any
from datetime import datetime
from numerize.numerize import numerize
from transformers import PreTrainedTokenizer


def init() -> None:
    warnings.filterwarnings(action="ignore")
    transformers.logging.set_verbosity_error()


def count_params(model: nn.Module) -> str:
    total_params, tunable_params = 0, 0
    for param in model.parameters():
        n_params = param.numel()
        if n_params == 0 and hasattr(param, "ds_numel"):
            n_params = param.ds_numel
        if param.__class__.__name__ == "Params4bit":
            n_params *= 2
        total_params += n_params
        if param.requires_grad:
            tunable_params += n_params
    cnt_str = " || ".join([f"Tunable Parameters: {numerize(tunable_params)}"
        , f"All Parameters: {numerize(total_params)}"
        , f"Tunable%: {tunable_params / total_params * 100:.3f}%"
    ])
    return cnt_str


def rank0_print(local_rank: int, *args) -> None:
    if local_rank == 0:
        print(*args)


def str_datetime() -> str:
    return datetime.now(pytz.timezone("Asia/Shanghai")).strftime("[%Y-%m-%d %H:%M:%S,%f")[:-3] + "]"


def split_list(lst: List, n: int) -> List[List[Any]]:
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst: List, n: int, k: int) -> List[Any]:
    chunks = split_list(lst, n)
    return chunks[k]