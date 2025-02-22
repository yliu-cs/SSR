import os
import re
import torch
import random
import autoroot
import depth_pro
import numpy as np
from PIL import Image
from typing import Tuple, Any
from functools import partial
from argparse import ArgumentParser
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from tqdm.contrib.concurrent import thread_map
from transformers import CLIPProcessor, SiglipVisionModel
from ssr.utils.prompt import SSRSpecialToken, string_truncation
from ssr.models.tokenization_internlm3 import Internlm3Tokenizer
from ssr.utils.misc import convert_depth, load_jsonl, get_chunk, change_ext


def parse_special_tokens(text: str) -> Tuple[str, str]:
    answer_match = re.search(r"<CONCLUSION>(.*?)</CONCLUSION>", text, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else None
    text_without_answer = re.sub(r"<CONCLUSION>.*?</CONCLUSION>", "", text, flags=re.DOTALL)
    rationale = re.sub(r"<\/?[A-Z]+>", "", text_without_answer)
    rationale = re.sub(r"\s+", " ", rationale).strip()
    if answer:
        answer = re.sub(r"\s+", " ", answer).strip()
    return rationale, answer


class LLaVACoTDataset(Dataset):
    def __init__(
        self
        , data_dir: str
        , tokenizer: Internlm3Tokenizer
        , max_length: Tuple[int, int, int]
        , clip_processor: CLIPProcessor
        , siglip_processor: SiglipVisionModel
    ) -> None:
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.clip_processor = clip_processor
        self.siglip_processor = siglip_processor
        self.data = load_jsonl(os.path.join(data_dir, "train.jsonl"))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        try:
            item = self.data[idx]
            image_path = os.path.join(self.data_dir, item["image"])
            image = Image.open(image_path).convert("RGB")
            conversations = item["conversations"]
            question, rationale, answer = None, None, None
            for conv in conversations:
                if conv["from"] == "human":
                    question = conv["value"]
                else:
                    rationale, answer = parse_special_tokens(conv["value"])
                if question and rationale and answer:
                    break
            question, rationale, answer = (string_truncation(text, self.tokenizer, max_len) for text, max_len in zip((question, rationale, answer), self.max_length))
            question = "\n".join([SSRSpecialToken.IMAGE_TOKEN, SSRSpecialToken.DEPTH_TOKEN, question])
            image = (self.clip_processor(images=image, return_tensors="pt").pixel_values).squeeze(0)
            depth_path = os.sep.join([self.data_dir] + [f"{item['image'].split(os.sep)[0]}_d"] + item["image"].split(os.sep)[1:])
            depth_path = change_ext(depth_path, "png")
            depth = convert_depth(np.array(Image.open(depth_path)), convert_16bits=True, convert_3channels=True)
            depth = (self.siglip_processor(images=depth, return_tensors="pt").pixel_values).squeeze(0)
            return {
                "question": question
                , "rationale": rationale
                , "answer": answer
                , "image": image
                , "depth": depth
            }
        except Exception as e:
            return random.choice(self)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "LLaVA-CoT-100k"))
    parser.add_argument("--model_path", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "DepthPro", "depth_pro.pt"))
    parser.add_argument("--num_chunk", type=int, default=8)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--max_workers", type=int, default=20)
    args = parser.parse_args()

    def save_image2depth(image_path: str, model: depth_pro.depth_pro.DepthPro, transform: Compose, device: torch.device) -> None:
        depth_path = os.sep.join([f"{image_path.split(os.sep)[0]}_d"] + image_path.split(os.sep)[1:])
        image_path, depth_path = (os.path.join(args.data_dir, path) for path in (image_path, depth_path))
        depth_path = change_ext(depth_path, "png")
        if os.path.exists(depth_path):
            return
        depth_dir = os.path.dirname(depth_path)
        if not os.path.exists(depth_dir):
            os.makedirs(depth_dir, exist_ok=True)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"{e=}")
            return
        image, _, f_px = depth_pro.load_pil(image)
        image = transform(image)
        image = image.to(device)
        depth = model.infer(image, f_px=f_px)["depth"]
        depth = convert_depth(depth, convert_16bits=True, convert_3channels=False)
        depth.save(depth_path)

    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    model, transform = depth_pro.create_model_and_transforms(checkpoint_uri=args.model_path, device=device)
    model.eval()

    llava_cot_data = load_jsonl(os.path.join(args.data_dir, "train.jsonl"))
    llava_cot_data = get_chunk(llava_cot_data, n=args.num_chunk, k=args.chunk_idx)
    llava_cot_data = list(map(lambda x: x["image"], llava_cot_data))
    thread_map(
        partial(
            save_image2depth
            , model=model
            , transform=transform
            , device=device
        )
        , llava_cot_data
        , max_workers=args.max_workers
        , desc=f"[{args.chunk_idx + 1}/{args.num_chunk}] Preprocess LLaVA CoT Dataset"
    )