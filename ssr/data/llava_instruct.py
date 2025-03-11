import os
import json
import torch
import autoroot
import depth_pro
import numpy as np
from PIL import Image
from itertools import chain
from functools import partial
from argparse import ArgumentParser
from typing import Dict, Any, Union, Tuple
from torchvision.transforms import Compose
from tqdm.contrib.concurrent import thread_map
from ssr.utils.misc import quiet, get_chunk, change_ext, freeze_module


if __name__ == "__main__":
    quiet()

    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "LLaVA-Instruct-150K"))
    parser.add_argument("--depthpro_path", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "DepthPro", "depth_pro.pt"))
    parser.add_argument("--max_workers", type=int, default=15)
    parser.add_argument("--num_chunks", type=int, default=8)
    parser.add_argument("--chunk_idx", type=int, default=0)
    args = parser.parse_args()

    def load_depth_pro(model_path: str, device: torch.device) -> Tuple[depth_pro.depth_pro.DepthPro, Compose]:
        model, transform = depth_pro.create_model_and_transforms(device=device, checkpoint_uri=model_path)
        freeze_module(model)
        model.eval()
        return model, transform

    def convert_depth(depth: Union[np.ndarray, torch.Tensor]) -> Image.Image:
        if isinstance(depth, torch.Tensor):
            depth = depth.squeeze().cpu().numpy()
        assert isinstance(depth, np.ndarray), "Depth must be a torch tensor or numpy array"
        assert depth.ndim == 2, "Depth must be 2D"
        depth = depth * 1000
        depth = depth.astype(np.uint16)
        depth = Image.fromarray(depth)
        return depth

    def save_image2depth(image_path: str, model: depth_pro.depth_pro.DepthPro, transform: Compose, device: torch.device) -> None:
        depth_dir = os.path.dirname(image_path)
        depth_dir = os.sep.join(depth_dir.split(os.sep)[:-1] + [f"{depth_dir.split(os.sep)[-1]}_d"])
        os.makedirs(depth_dir, exist_ok=True)
        depth_path = change_ext(os.path.join(depth_dir, os.path.basename(image_path)), "png")
        if os.path.exists(depth_path):
            return
        image = Image.open(image_path).convert("RGB")
        image, _, f_px = depth_pro.load_pil(image)
        image = transform(image)
        image = image.to(device)
        depth = model.infer(image, f_px=f_px)["depth"]
        depth = convert_depth(depth)
        depth.save(depth_path)

    def preprocess_item(item: str, depthpro: depth_pro.depth_pro.DepthPro, transform: Compose, device: torch.device) -> Dict[str, Any]:
        try:
            if "image" not in item:
                return []
            data = []
            image_path = item["image"]
            image_path = image_path.replace("vg/VG_100K_2", os.path.join("LLaVA-Instruct-150K", "VisualGenome"))
            image_path = image_path.replace("vg/VG_100K", os.path.join("LLaVA-Instruct-150K", "VisualGenome"))
            image_path = image_path.replace("ocr_vqa/images", os.path.join("LLaVA-Instruct-150K", "OCR-VQA"))
            image_path = image_path.replace("gqa/images", os.path.join("VoCoT", "images", "GQA"))
            image_path = image_path.replace("coco/train2017", os.path.join("VoCoT", "images", "COCO"))
            image_path = image_path.replace("textvqa/train_images", os.path.join("LLaVA-Instruct-150K", "TextVQA"))
            if "COCO" in image_path:
                image_path = os.path.join(os.path.dirname(image_path), f"COCO_train2014_{os.path.basename(image_path)}")
            if not os.path.exists(os.path.join(os.path.dirname(args.data_dir), image_path)):
                return []
            save_image2depth(os.path.join(os.path.dirname(args.data_dir), image_path), depthpro, transform, device)
            for conv_idx in range(0, len(item["conversations"]), 2):
                if item["conversations"][conv_idx]["from"] != "human" or item["conversations"][conv_idx + 1]["from"] != "gpt":
                    continue
                question, answer = item["conversations"][conv_idx]["value"], item["conversations"][conv_idx + 1]["value"]
                question = question.replace("<image>", "").strip("\n").strip()
                data.append({
                    "image_path": image_path
                    , "question": question
                    , "answer": answer
                })
            return data
        except:
            return []

    raw_data = json.load(open(os.path.join(args.data_dir, "llava_v1_5_mix665k.json")))
    raw_data = get_chunk(raw_data, args.num_chunks, args.chunk_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depthpro, transform = load_depth_pro(args.depthpro_path, device)

    data = thread_map(
        partial(preprocess_item, depthpro=depthpro, transform=transform, device=device)
        , raw_data
        , max_workers=args.max_workers
        , desc=f"[Chunk {args.chunk_idx}/{args.num_chunks}] Preprocessing data"
        , ncols=100
    )
    data = list(chain(*data))
    np.save(os.path.join(args.data_dir, f"ssr_llava_instruct_{args.chunk_idx}_{args.num_chunks}.npy"), data)