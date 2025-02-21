import os
import json
import torch
import random
import autoroot
import depth_pro
import numpy as np
from PIL import Image
from typing import Any
from functools import partial
from argparse import ArgumentParser
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from ssr.utils.prompt import SSRSpecialToken
from tqdm.contrib.concurrent import thread_map
from transformers import CLIPProcessor, SiglipVisionModel
from ssr.utils.misc import convert_depth, get_chunk, change_ext, load_jsonl


class VoCoTDataset(Dataset):
    def __init__(
        self
        , data_dir: str
        , clip_processor: CLIPProcessor
        , siglip_processor: SiglipVisionModel
    ) -> None:
        self.data_dir = data_dir
        self.clip_processor = clip_processor
        self.siglip_processor = siglip_processor
        self.data = load_jsonl(os.path.join(data_dir, "ssr_vocot.jsonl"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        try:
            item = self.data[idx]
            question, rationale, answer, image_path = item["question"], item["rationale"], item["answer"], item["image_path"]
            question = "\n".join([SSRSpecialToken.IMAGE_TOKEN, SSRSpecialToken.DEPTH_TOKEN, question])
            depth_path = os.sep.join([f"{item['image_path'].split(os.sep)[0]}_d"] + item["image_path"].split(os.sep)[1:])
            depth_path = change_ext(depth_path, "png")
            image_path, depth_path = [os.path.join(self.data_dir, "images", key) for key in (image_path, depth_path)]
            image = Image.open(image_path).convert("RGB")
            image = (self.clip_processor(images=image, return_tensors="pt").pixel_values).squeeze(0)
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
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "VoCoT"))
    parser.add_argument("--model_path", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "DepthPro", "depth_pro.pt"))
    parser.add_argument("--num_chunk", type=int, default=16)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--max_workers", type=int, default=20)
    args = parser.parse_args()

    def save_image2depth(image_path: str, model: depth_pro.depth_pro.DepthPro, transform: Compose, device: torch.device) -> None:
        depth_path = os.sep.join([f"{image_path.split(os.sep)[0]}_d"] + image_path.split(os.sep)[1:])
        image_path, depth_path = (os.path.join(args.data_dir, "images", path) for path in (image_path, depth_path))
        depth_path = change_ext(depth_path, "png")
        if os.path.exists(depth_path):
            return
        depth_dir = os.path.dirname(depth_path)
        if not os.path.exists(depth_dir):
            os.makedirs(depth_dir, exist_ok=True)
        image = Image.open(image_path).convert("RGB")
        image, _, f_px = depth_pro.load_pil(image)
        image = transform(image)
        image = image.to(device)
        depth = model.infer(image, f_px=f_px)["depth"]
        depth = convert_depth(depth, convert_16bits=True, convert_3channels=False)
        depth.save(depth_path)

    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    model, transform = depth_pro.create_model_and_transforms(checkpoint_uri=args.model_path, device=device)
    model.eval()

    with open(os.path.join(args.data_dir, "VoCoT-80K_integrated.json"), "r", encoding="utf-8") as file:
        vo_cot_data = json.load(file)
    vo_cot_data = get_chunk(vo_cot_data, n=args.num_chunk, k=args.chunk_idx)
    vo_cot_data = list(map(lambda x: x["image"], vo_cot_data))
    vo_cot_data = list(map(lambda x: x.replace(os.path.join("COCO2015", "images", "train2014"), "COCO"), vo_cot_data))
    vo_cot_data = list(map(lambda x: x.replace(os.path.join("LVIS", "train2017"), "LVIS"), vo_cot_data))
    thread_map(
        partial(
            save_image2depth
            , model=model
            , transform=transform
            , device=device
        )
        , vo_cot_data
        , max_workers=args.max_workers
        , desc=f"[{args.chunk_idx + 1}/{args.num_chunk}] Preprocess VoCoT Dataset"
    )