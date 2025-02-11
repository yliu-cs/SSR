import os
import torch
from typing import List
from functools import partial
from dataclasses import dataclass
from datasets import load_dataset
from depth_pro.depth_pro import DepthPro
from torch.utils.data import ChainDataset
from torchvision.transforms import Compose
from ssr.data.llava_cot import LLaVACoTDataset
from transformers import CLIPProcessor, SiglipVisionModel
from ssr.models.tokenization_internlm3 import Internlm3Tokenizer
from ssr.utils.prompt import SSRStage, SSRSpecialToken, repeat_special_tokens, construct_conversation, create_labels


def prepare_ssr_dataset(
    cot_data_dirs: List[str]
    , clip_processor: CLIPProcessor
    , siglip_processor: SiglipVisionModel
) -> ChainDataset:
    dataset_map = {
        "LLaVA-CoT-100k": LLaVACoTDataset
    }
    datasets = []
    for cot_data_dir in cot_data_dirs:
        datasets.append(
            dataset_map[os.path.basename(cot_data_dir)](
                data_dir=cot_data_dir
                , clip_processor=clip_processor
                , siglip_processor=siglip_processor
            )
        )
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