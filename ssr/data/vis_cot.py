import os
import torch
import autoroot
import depth_pro
from PIL import Image
from itertools import chain
from functools import partial
from argparse import ArgumentParser
from torchvision.transforms import Compose
from tqdm.contrib.concurrent import thread_map
from ssr.utils.misc import convert_depth, get_chunk, change_ext, load_jsonl


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "Visual-CoT"))
    parser.add_argument("--model_path", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "DepthPro", "depth_pro.pt"))
    parser.add_argument("--num_chunk", type=int, default=16)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--max_workers", type=int, default=15)
    args = parser.parse_args()

    def save_image2depth(image_path: str, model: depth_pro.depth_pro.DepthPro, transform: Compose, device: torch.device) -> None:
        depth_path = os.sep.join([f"{image_path.split(os.sep)[0]}_d"] + image_path.split(os.sep)[1:])
        depth_path = os.path.join(args.data_dir, "cot_image_data", depth_path)
        depth_path = change_ext(depth_path, "png")
        if os.path.exists(depth_path):
            return
        depth_dir = os.path.dirname(depth_path)
        if not os.path.exists(depth_dir):
            os.makedirs(depth_dir, exist_ok=True)
        image = Image.open(os.path.join(args.data_dir, "cot_image_data", image_path)).convert("RGB")
        image, _, f_px = depth_pro.load_pil(image)
        image = transform(image)
        image = image.to(device)
        depth = model.infer(image, f_px=f_px)["depth"]
        depth = convert_depth(depth, convert_16bits=True, convert_3channels=False)
        depth.save(depth_path)

    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    model, transform = depth_pro.create_model_and_transforms(checkpoint_uri=args.model_path, device=device)
    model.eval()

    vis_cot_data = list(chain(*[load_jsonl(os.path.join(args.data_dir, "cot_data", filename)) for filename in os.listdir(os.path.join(args.data_dir, "cot_data"))]))
    vis_cot_data = get_chunk(vis_cot_data, n=args.num_chunk, k=args.chunk_idx)
    vis_cot_data = list(map(lambda x: os.path.join(x["dataset"], x["image"]), vis_cot_data))[:10]
    thread_map(
        partial(
            save_image2depth
            , model=model
            , transform=transform
            , device=device
        )
        , vis_cot_data
        , max_workers=args.max_workers
        , desc=f"[{args.chunk_idx + 1}/{args.num_chunk}] Preprocess Visual CoT Dataset"
    )