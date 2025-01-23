import re
import pytz
import math
import torch
import warnings
import numpy as np
import transformers
from torch import nn
from PIL import Image
from datetime import datetime
from typing import List, Any, Union
from numerize.numerize import numerize


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


def convert_depth(depth: Union[np.ndarray, torch.Tensor], convert_16bits: bool = True):
    if convert_16bits:
        if isinstance(depth, torch.Tensor):
            depth = depth.squeeze().cpu().numpy()
        assert isinstance(depth, np.ndarray), "Depth must be a torch tensor or numpy array"
        assert depth.ndim == 2, "Depth must be 2D"
        depth = depth * 1000
        depth = depth.astype(np.uint16)
        depth = Image.fromarray(depth)
    channels = len(depth.getbands())
    if channels == 1:
        img = np.array(depth)
        height, width = img.shape
        three_channel_array = np.zeros((height, width, 3), dtype=np.uint8)
        three_channel_array[:, :, 0] = (img // 1024) * 4
        three_channel_array[:, :, 1] = (img // 32) * 8
        three_channel_array[:, :, 2] = (img % 32) * 8
        depth = Image.fromarray(three_channel_array, "RGB")
    return depth

    
def build_projector(mm_hidden_size: int = 1024, hidden_size: int = 4096) -> nn.Sequential:
    projector_type = "mlp2x_gelu"
    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(mm_hidden_size, hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        return nn.Sequential(*modules)
    raise ValueError(f"Unknown projector type: {projector_type}")


def freeze_module(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def has_nan(tensor: torch.Tensor) -> bool:
    return torch.isnan(tensor).any()


def get_grad(model: nn.Module) -> float:
    grad = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            grad += param_norm.item() ** 2
    grad = grad ** 0.5
    return grad