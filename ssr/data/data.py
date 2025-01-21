import torch
from PIL import Image
from functools import partial
from depth_pro import load_pil
from dataclasses import dataclass
from datasets import load_dataset
from typing import List, Dict, Any
from depth_pro.depth_pro import DepthPro
from torch.utils.data import ChainDataset
from torchvision.transforms import Compose
from transformers import CLIPProcessor, SiglipVisionModel


def load_depth(
    image: Image
    , depth_pro: DepthPro
    , depth_transform: Compose
):
    image, _, f_px = load_pil(image)
    image = depth_transform(image)
    depth = (depth_pro.infer(image.to(next(depth_pro.parameters()).device), f_px=f_px))["depth"]
    return depth


def prepare_vrc(
    raw_data: Dict[str, Any]
    , clip_processor: CLIPProcessor
    , siglip_processor: SiglipVisionModel
    , depth_pro: DepthPro
    , depth_transform: Compose
) -> Dict[str, Any]:
    image = raw_data["image"].convert("RGB")
    question = raw_data["question"]
    rationale = "".join(raw_data["steps"])
    answer = raw_data["final_answer"]
    return {
        "question": "<image>\n" + question
        , "rationale": rationale
        , "answer": answer
        , "image": (clip_processor(images=image, return_tensors="pt").pixel_values).squeeze(0)
        # , "depth": (siglip_processor(images=raw_data["depth"].convert("RGB"), return_tensors="pt").pixel_values).squeeze(0)
        , "depth": load_depth(image=image, depth_pro=depth_pro, depth_transform=depth_transform)
    }


def prepare_ssr_dataset(
    cot_data_names: List[str]
    , clip_processor: CLIPProcessor
    , siglip_processor: SiglipVisionModel
    , depth_pro: DepthPro
    , depth_transform: Compose
) -> ChainDataset:
    data_pre_func_map = {
        "vrc-bench": partial(
            prepare_vrc
            , clip_processor=clip_processor
            , siglip_processor=siglip_processor
            , depth_pro=depth_pro
            , depth_transform=depth_transform
        )
    }
    datasets = [load_dataset(cot_data_name, split="test", streaming=True) for cot_data_name in cot_data_names]
    for i in range(len(datasets)):
        datasets[i] = datasets[i].map(
            data_pre_func_map[datasets[i]._info.dataset_name]
            , remove_columns=list(set(list(datasets[i].features.keys())) - set(["question", "rationale", "answer", "image", "depth"]))
        ).with_format("torch")
    dataset = ChainDataset(datasets)
    return dataset


@dataclass
class SSRDataCollator(object):
    def __call__(self, instances: list[dict]) -> dict[str, torch.Tensor]:
        for i in range(len(instances)):
            print(f"{instances[i].keys()=}")
        question, rationale, answer, image, depth = tuple([instance[key] for instance in instances] for key in ("question", "rationale", "answer", "image", "depth"))
        print(f"{type(question[0])=}")
        print(f"{type(rationale[0])=}")
        print(f"{type(answer[0])=}")
        print(f"{type(image[0])=}")
        print(f"{type(depth[0])=}")
        exit()