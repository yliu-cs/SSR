import os
import torch
import autoroot
import numpy as np
from torch import nn
from PIL import Image
from typing import Tuple
from torch.utils.data import Dataset
from ssr.utils.misc import load_jsonl, colorize_depth
from ssr.utils.prompt import IGNORE_INDEX, SSRSpecialToken, insert_tor, string_truncation
from transformers import PreTrainedTokenizer, CLIPProcessor, CLIPVisionModel, SiglipProcessor, SiglipVisionModel


def get_visual_embeds(
    raw_image: np.ndarray
    , raw_depth: np.ndarray
    , clip_processor: CLIPProcessor
    , clip_model: CLIPVisionModel
    , siglip_processor: SiglipProcessor
    , siglip_model: SiglipVisionModel
) -> Tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        image_embeds = (clip_model(**(clip_processor(images=raw_image, return_tensors="pt").to(clip_model.device))).last_hidden_state).squeeze(0).detach()
        depth_embeds = (siglip_model(**(siglip_processor(images=raw_depth, return_tensors="pt").to(siglip_model.device))).last_hidden_state).squeeze(0).detach()
    return image_embeds, depth_embeds


class SSRCoTDataset4Reasoning(Dataset):
    def __init__(
        self
        , data_dir: str
        , n_tor: int
        , mamba_tokenizer: PreTrainedTokenizer
        , llm_tokenizer: PreTrainedTokenizer
        , max_length: Tuple[int, int, int]
        , clip_processor: CLIPProcessor
        , clip_model: CLIPVisionModel
        , siglip_processor: SiglipProcessor
        , siglip_model: SiglipVisionModel
    ) -> None:
        self.data_dir = data_dir
        self.n_tor = n_tor
        self.data = load_jsonl(os.path.join(data_dir, "ssr-cot.jsonl"))
        self.mamba_tokenizer = mamba_tokenizer
        self.llm_tokenizer = llm_tokenizer
        self.max_length = max_length
        self.clip_processor = clip_processor
        self.clip_model = clip_model
        self.siglip_processor = siglip_processor
        self.siglip_model = siglip_model
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> dict:
        data = self.data[index]
        question, rationale, answer = data["question"], data["rationale"], data["answer"]
        question, rationale, answer = [string_truncation(text, self.mamba_tokenizer, max_len) for text, max_len in zip((question, rationale, answer), self.max_length)]
        rationale = insert_tor(rationale, n_tor=self.n_tor)
        mamba_question = self.mamba_tokenizer(question, add_special_tokens=False, return_tensors="pt")
        mamba_rationale = self.mamba_tokenizer(rationale, add_special_tokens=False, return_tensors="pt")
        mamba_input_ids = torch.cat((mamba_question.input_ids, mamba_rationale.input_ids), dim=1).squeeze(0)
        mamba_attention_mask = torch.cat((mamba_question.attention_mask, mamba_rationale.attention_mask), dim=1).squeeze(0)
        llm_rationale = self.llm_tokenizer(rationale, add_special_tokens=False, return_tensors="pt")
        llm_input_ids, llm_attention_mask = llm_rationale.input_ids.squeeze(0), llm_rationale.attention_mask.squeeze(0)
        llm_labels = llm_input_ids.clone()
        llm_labels[llm_input_ids == self.llm_tokenizer._tokenizer.token_to_id(SSRSpecialToken.TOR_TOKEN)] = IGNORE_INDEX
        raw_image = np.array(Image.open(os.path.join(self.data_dir, data["image_path"])).convert("RGB"))
        raw_depth = colorize_depth(os.path.join(self.data_dir, data["depth_path"]))
        image_embeds, depth_embeds = get_visual_embeds(raw_image, raw_depth, self.clip_processor, self.clip_model, self.siglip_processor, self.siglip_model)
        mamba_attention_mask = torch.cat((torch.ones(image_embeds.size(0) + depth_embeds.size(0), dtype=torch.long), mamba_attention_mask))
        mamba_labels = torch.cat((
            torch.full((image_embeds.size(0) + depth_embeds.size(0) + mamba_question.input_ids.size(1),), IGNORE_INDEX, dtype=torch.long)
            , mamba_rationale.input_ids.squeeze(0)
        ))
        return {
            "mamba_input_ids": mamba_input_ids
            , "mamba_attention_mask": mamba_attention_mask
            , "mamba_labels": mamba_labels
            , "llm_input_ids": llm_input_ids
            , "llm_attention_mask": llm_attention_mask
            , "llm_labels": llm_labels
            , "image_embeds": image_embeds
            , "depth_embeds": depth_embeds
        }
    
    def collate_fn(self, batch: list[dict]) -> dict:
        mamba_input_ids, mamba_attention_mask, mamba_labels = [[item[key] for item in batch] for key in ("mamba_input_ids", "mamba_attention_mask", "mamba_labels")]
        llm_input_ids, llm_attention_mask, llm_labels = [[item[key] for item in batch] for key in ("llm_input_ids", "llm_attention_mask", "llm_labels")]
        mamba_input_ids = nn.utils.rnn.pad_sequence(sequences=mamba_input_ids, batch_first=True, padding_value=self.mamba_tokenizer.pad_token_id, padding_side="left")
        mamba_attention_mask = nn.utils.rnn.pad_sequence(sequences=mamba_attention_mask, batch_first=True, padding_value=0, padding_side="left")
        mamba_labels = nn.utils.rnn.pad_sequence(sequences=mamba_labels, batch_first=True, padding_value=IGNORE_INDEX, padding_side="left")
        llm_input_ids = nn.utils.rnn.pad_sequence(sequences=llm_input_ids, batch_first=True, padding_value=self.llm_tokenizer.pad_token_id, padding_side="left")
        llm_attention_mask = nn.utils.rnn.pad_sequence(sequences=llm_attention_mask, batch_first=True, padding_value=0, padding_side="left")
        llm_labels = nn.utils.rnn.pad_sequence(sequences=llm_labels, batch_first=True, padding_value=IGNORE_INDEX, padding_side="left")
        image_embeds, depth_embeds = [torch.stack([item[key] for item in batch]) for key in ("image_embeds", "depth_embeds")]
        return {
            "mamba_input_ids": mamba_input_ids
            , "mamba_attention_mask": mamba_attention_mask
            , "mamba_labels": mamba_labels
            , "llm_input_ids": llm_input_ids
            , "llm_attention_mask": llm_attention_mask
            , "llm_labels": llm_labels
            , "image_embeds": image_embeds
            , "depth_embeds": depth_embeds
        }


if __name__ == "__main__":
    from ssr.utils.misc import quiet
    from transformers import AutoTokenizer
    quiet()

    mamba_tokenizer = AutoTokenizer.from_pretrained("/ssdwork/liuyang/Models/mamba-130m-hf")
    mamba_tokenizer.add_tokens(SSRSpecialToken.TOR_TOKEN, special_tokens=True)
    llm_tokenizer = AutoTokenizer.from_pretrained("/ssdwork/liuyang/Models/Qwen2.5-3B")
    llm_tokenizer.add_tokens(SSRSpecialToken.TOR_TOKEN, special_tokens=True)
    dataset = SSRCoTDataset4Reasoning(
        data_dir=os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "SSR-CoT")
        , n_tor=10
        , mamba_tokenizer=mamba_tokenizer
        , llm_tokenizer=llm_tokenizer
        , max_length=(128, 1024, 128)
    )
    data = dataset[0]
    for key, value in data.items():
        print(f"{key}: {value.size() if isinstance(value, torch.Tensor) else value}")