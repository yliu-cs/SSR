import torch
from PIL import Image
from functools import partial
from depth_pro import load_pil
from dataclasses import dataclass
from datasets import load_dataset
from typing import List, Dict, Any
from ssr.utils.misc import convert_depth
from depth_pro.depth_pro import DepthPro
from torch.utils.data import ChainDataset
from torchvision.transforms import Compose
from transformers import CLIPProcessor, SiglipVisionModel
from ssr.models.tokenization_internlm3 import Internlm3Tokenizer
from ssr.utils.prompt import SSRStage, SSRSpecialToken, repeat_special_tokens, construct_conversation, create_labels


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
        "question": "\n".join([SSRSpecialToken.IMAGE_TOKEN, SSRSpecialToken.DEPTH_TOKEN, question])
        , "rationale": rationale
        , "answer": answer
        , "image": (
            clip_processor(
                images=image
                , return_tensors="pt"
            ).pixel_values
        ).squeeze(0)
        , "depth": (
            siglip_processor(
                images=convert_depth(
                    load_depth(image=image, depth_pro=depth_pro, depth_transform=depth_transform)
                    , convert_16bits=True
                )
                , return_tensors="pt"
            ).pixel_values
        ).squeeze(0)
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
    stage: SSRStage
    n_tor: int
    n_image_tokens: int
    n_depth_tokens: int
    tokenizer: Internlm3Tokenizer
    def __call__(self, instances: list[dict]) -> dict[str, torch.Tensor]:
        convs = []
        for instance in instances:
            question, rationale, answer = (instance[key] for key in ("question", "rationale", "answer"))
            conv = repeat_special_tokens(
                input_string=construct_conversation(
                    question=question
                    , rationale=rationale if self.stage == SSRStage.mamba else ""
                    , answer=answer
                    , stage=self.stage
                    , n_tor=self.n_tor
                )
                , special_tokens=[SSRSpecialToken.IMAGE_TOKEN, SSRSpecialToken.DEPTH_TOKEN]
                , n_repeats=[self.n_image_tokens, self.n_depth_tokens]
            )
            convs.append(conv)
        inputs = self.tokenizer(convs, padding="longest", return_tensors="pt", add_special_tokens=False)
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        labels = create_labels(input_ids=input_ids, stage=self.stage, tokenizer=self.tokenizer)
        image, depth = tuple([instance[key] for instance in instances] for key in ("image", "depth"))
        image = torch.stack(image, dim=0)
        depth = torch.stack(depth, dim=0)
        return {
            "input_ids": input_ids
            , "attention_mask": attention_mask
            , "labels": labels
            , "image": image
            , "depth": depth
        }