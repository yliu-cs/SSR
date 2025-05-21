import os
import torch
import autoroot
import numpy as np
from torch import nn
from PIL import Image
from random import choice
from typing import List, Tuple, Dict
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info
from ssr.utils.misc import load_jsonl, colorize_depth
from ssr.utils.prompt import IGNORE_INDEX, SSRSpecialToken, insert_tor, string_truncation
from transformers import PreTrainedTokenizer, CLIPProcessor, CLIPVisionModel, SiglipProcessor, SiglipVisionModel, Qwen2_5_VLProcessor


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
        try:
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
            mamba_attention_mask = torch.cat((torch.ones(image_embeds.size(0) + depth_embeds.size(0), dtype=mamba_attention_mask.dtype), mamba_attention_mask))
            mamba_labels = torch.cat((
                torch.full((image_embeds.size(0) + depth_embeds.size(0) + mamba_question.input_ids.size(1),), IGNORE_INDEX, dtype=mamba_rationale.input_ids.dtype)
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
        except Exception as e:
            print(f"{e=}")
            return choice(self)
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
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


class SSRCoTDataset4VLM(Dataset):
    def __init__(
        self
        , data_dir: str
        , n_tor: int
        , mamba_tokenizer: PreTrainedTokenizer
        , vlm_processor: Qwen2_5_VLProcessor
        , max_length: Tuple[int, int, int]
        , image_size: Tuple[int, int]
        , clip_processor: CLIPProcessor
        , clip_model: CLIPVisionModel
        , siglip_processor: SiglipProcessor
        , siglip_model: SiglipVisionModel
        , llava: bool = True
    ) -> None:
        self.data_dir = data_dir
        self.n_tor = n_tor
        self.mamba_tokenizer = mamba_tokenizer
        self.vlm_processor = vlm_processor
        self.max_length = max_length[:1] + max_length[-1:]
        self.image_size = image_size
        self.clip_processor = clip_processor
        self.clip_model = clip_model
        self.siglip_processor = siglip_processor
        self.siglip_model = siglip_model
        self.data = load_jsonl(os.path.join(self.data_dir, "ssr-cot.jsonl"))
        if llava:
            self.data += load_jsonl(os.path.join(self.data_dir, "LLaVA-Instruct-150K", "ssr_llava_inst.jsonl"))
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> dict:
        try:
            item = self.data[index]
            question, answer, image_path, depth_path = item["question"], item["answer"], item["image_path"], item["depth_path"]
            question, answer = [string_truncation(text, self.mamba_tokenizer, max_len) for text, max_len in zip((question, answer), self.max_length)]
            rationale = insert_tor("", n_tor=self.n_tor)
            mamba_question = self.mamba_tokenizer(question, add_special_tokens=False, return_tensors="pt")
            mamba_rationale = self.mamba_tokenizer(rationale, add_special_tokens=False, return_tensors="pt")
            mamba_input_ids = torch.cat((mamba_question.input_ids, mamba_rationale.input_ids), dim=1).squeeze(0)
            mamba_attention_mask = torch.cat((mamba_question.attention_mask, mamba_rationale.attention_mask), dim=1).squeeze(0)
            image_path, depth_path = os.path.join(self.data_dir, image_path), os.path.join(self.data_dir, depth_path)
            raw_image = Image.open(image_path).convert("RGB")
            raw_depth = colorize_depth(depth_path)
            image_embeds, depth_embeds = get_visual_embeds(np.array(raw_image), raw_depth, self.clip_processor, self.clip_model, self.siglip_processor, self.siglip_model)
            mamba_attention_mask = torch.cat((torch.ones(image_embeds.size(0) + depth_embeds.size(0), dtype=mamba_attention_mask.dtype), mamba_attention_mask))
            messages = [
                {
                    "role": "user"
                    , "content": [
                        {"type": "image", "image": raw_image.resize(self.image_size)}
                        , {"type": "text", "text": f"{rationale}\n{question}"}
                    ]
                }
                , {
                    "role": "assistant"
                    , "content": [{"type": "text", "text": answer}]
                }
            ]
            text = self.vlm_processor.apply_chat_template(messages, tokenize=False)
            image_inputs, _ = process_vision_info(messages)
            inputs = self.vlm_processor(text=[text], images=image_inputs, videos=None, padding=False, return_tensors="pt")
            vlm_input_ids = inputs.input_ids.squeeze(0)
            vlm_attention_mask = inputs.attention_mask.squeeze(0)
            vlm_pixel_values = inputs.pixel_values.squeeze(0)
            vlm_image_grid_thw = inputs.image_grid_thw.squeeze(0)
            vlm_labels = torch.full_like(vlm_input_ids, IGNORE_INDEX, dtype=vlm_input_ids.dtype)
            start_idx = 0
            for i in range(vlm_input_ids.size(0)):
                if vlm_input_ids[i:i + 2].equal(torch.tensor(self.vlm_processor.tokenizer.convert_tokens_to_ids(["<|im_start|>", "assistant"]), dtype=vlm_input_ids.dtype)):
                    start_idx = i
                    break
            assert start_idx > 1
            start_idx -= 1
            vlm_labels[start_idx:] = vlm_input_ids[start_idx:].clone()
            return {
                "mamba_input_ids": mamba_input_ids
                , "mamba_attention_mask": mamba_attention_mask
                , "image_embeds": image_embeds
                , "depth_embeds": depth_embeds
                , "vlm_input_ids": vlm_input_ids
                , "vlm_attention_mask": vlm_attention_mask
                , "vlm_pixel_values": vlm_pixel_values
                , "vlm_image_grid_thw": vlm_image_grid_thw
                , "vlm_labels": vlm_labels
            }
        except Exception as e:
            print(f"{e=}")
            return choice(self)
    
    def collate_fn(self, batch: list[dict]) -> Dict[str, torch.Tensor]:
        mamba_input_ids, mamba_attention_mask = [[item[key] for item in batch] for key in ("mamba_input_ids", "mamba_attention_mask")]
        image_embeds, depth_embeds = [torch.stack([item[key] for item in batch]) for key in ("image_embeds", "depth_embeds")]
        vlm_input_ids, vlm_attention_mask, vlm_labels = [[item[key] for item in batch] for key in ("vlm_input_ids", "vlm_attention_mask", "vlm_labels")]
        vlm_pixel_values, vlm_image_grid_thw = [torch.stack([item[key] for item in batch]) for key in ("vlm_pixel_values", "vlm_image_grid_thw")]
        mamba_input_ids = nn.utils.rnn.pad_sequence(sequences=mamba_input_ids, batch_first=True, padding_value=self.mamba_tokenizer.pad_token_id, padding_side="left")
        mamba_attention_mask = nn.utils.rnn.pad_sequence(sequences=mamba_attention_mask, batch_first=True, padding_value=0, padding_side="left")
        vlm_input_ids = nn.utils.rnn.pad_sequence(sequences=vlm_input_ids, batch_first=True, padding_value=self.vlm_processor.tokenizer.pad_token_id, padding_side="left")
        vlm_attention_mask = nn.utils.rnn.pad_sequence(sequences=vlm_attention_mask, batch_first=True, padding_value=0, padding_side="left")
        vlm_labels = nn.utils.rnn.pad_sequence(sequences=vlm_labels, batch_first=True, padding_value=IGNORE_INDEX, padding_side="left")
        return {
            "mamba_input_ids": mamba_input_ids
            , "mamba_attention_mask": mamba_attention_mask
            , "image_embeds": image_embeds
            , "depth_embeds": depth_embeds
            , "vlm_input_ids": vlm_input_ids
            , "vlm_attention_mask": vlm_attention_mask
            , "vlm_pixel_values": vlm_pixel_values
            , "vlm_image_grid_thw": vlm_image_grid_thw
            , "vlm_labels": vlm_labels
        }


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from ssr.utils.misc import quiet, str_datetime
    quiet()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{str_datetime()} Loading Tokenizer...")
    mamba_tokenizer = AutoTokenizer.from_pretrained("/ssdwork/liuyang/Models/mamba-130m-hf")
    mamba_tokenizer.add_tokens(SSRSpecialToken.TOR_TOKEN, special_tokens=True)
    llm_tokenizer = AutoTokenizer.from_pretrained("/ssdwork/liuyang/Models/Qwen2.5-3B")
    llm_tokenizer.add_tokens(SSRSpecialToken.TOR_TOKEN, special_tokens=True)

    print(f"{str_datetime()} Loading VLM Processor...")
    vlm_processor = Qwen2_5_VLProcessor.from_pretrained("/ssdwork/liuyang/Models/Qwen2.5-VL-3B-Instruct")
    vlm_processor.tokenizer.add_tokens(SSRSpecialToken.TOR_TOKEN, special_tokens=True)

    print(f"{str_datetime()} Loading CLIP Processor & Model...")
    clip_processor = CLIPProcessor.from_pretrained("/ssdwork/liuyang/Models/clip-vit-large-patch14-336")
    clip_model = (CLIPVisionModel.from_pretrained("/ssdwork/liuyang/Models/clip-vit-large-patch14-336")).to(device)

    print(f"{str_datetime()} Loading Siglip Processor & Model...")
    siglip_processor = SiglipProcessor.from_pretrained("/ssdwork/liuyang/Models/siglip-so400m-patch14-384")
    siglip_model = (SiglipVisionModel.from_pretrained("/ssdwork/liuyang/Models/siglip-so400m-patch14-384")).to(device)

    # dataset = SSRCoTDataset4Reasoning(
    #     data_dir=os.path.join(os.sep, "ssdwork", "liuyang", "Dataset")
    #     , n_tor=10
    #     , mamba_tokenizer=mamba_tokenizer
    #     , llm_tokenizer=llm_tokenizer
    #     , max_length=(128, 1024, 128)
    #     , clip_processor=clip_processor
    #     , clip_model=clip_model
    #     , siglip_processor=siglip_processor
    #     , siglip_model=siglip_model
    # )
    # data = dataset[0]
    # for key, value in data.items():
    #     print(f"{key}: {value.size() if isinstance(value, torch.Tensor) else value}")
    
    dataset = SSRCoTDataset4VLM(
        data_dir=os.path.join(os.sep, "ssdwork", "liuyang", "Dataset")
        , n_tor=10
        , mamba_tokenizer=mamba_tokenizer
        , vlm_processor=vlm_processor
        , max_length=(128, 1024, 128)
        , clip_processor=clip_processor
        , clip_model=clip_model
        , siglip_processor=siglip_processor
        , siglip_model=siglip_model
    )
    data = dataset[0]
    for key, value in data.items():
        print(f"{key}: {value.size() if isinstance(value, torch.Tensor) else value}")