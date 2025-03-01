import os
import json
import torch
import argparse
import autoroot
import depth_pro
import numpy as np
from PIL import Image
from tqdm import tqdm
from depth_pro.depth_pro import DepthPro
from typing import List, Dict, Any, Tuple
from torchvision.transforms import Compose
from ssr.models.modeling_ssr import SSRConfig, SSR
from transformers import CLIPProcessor, SiglipProcessor
from ssr.models.tokenization_internlm3 import Internlm3Tokenizer
from ssr.utils.misc import init, str_datetime, convert_depth, change_ext
from ssr.utils.load_ptm import load_clip_vit, load_siglip, load_depth_pro
from ssr.utils.prompt import SSRSpecialToken, construct_conversation, repeat_special_tokens


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Checkpoint", "SSR", "stage2"))
    parser.add_argument("--clip_path", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "clip-vit-large-patch14-336"))
    parser.add_argument("--siglip_path", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "siglip-so400m-patch14-384"))
    parser.add_argument("--depthpro_path", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "DepthPro"))
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "SpatialBench"))
    parser.add_argument("--task", type=str, default="size", choices=["size", "positional", "reach", "existence", "counting"])
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.2)
    return parser.parse_args()


def get_depth(image_path: str, depthpro: DepthPro, depth_transform: Compose) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image, _, f_px = depth_pro.load_pil(image)
    image = depth_transform(image)
    image = image.to("cuda")
    prediction = depthpro.infer(image, f_px=f_px)
    depth = prediction["depth"]
    return depth


def load_ptm(
    pretrained_path: str
    , clip_path: str
    , siglip_path: str
    , depthpro_path: str
    , device: torch.device
) -> SSR:
    ssr_config = SSRConfig.from_pretrained(pretrained_path)
    ssr_config.stage += 1
    ssr_config.stage2_path = pretrained_path
    print(f"{str_datetime()} Loading CLIPVisionModel ...")
    clip_processor, clip_vision = load_clip_vit(clip_path)
    print(f"{str_datetime()} Loading SigLIP ...")
    siglip_processor, siglip = load_siglip(siglip_path)
    print(f"{str_datetime()} Loading DepthPro ...")
    depthpro, depth_transform = load_depth_pro(depthpro_path, device=device)
    print(f"{str_datetime()} Loading SSR ...")
    ssr = SSR(ssr_config, clip_vision=clip_vision, siglip=siglip)
    ssr.prepare_modules()
    ssr = ssr.to(device=device)
    print(f"{str_datetime()} Loading Model Done !")
    return (clip_processor, clip_vision), (siglip_processor, siglip), (depthpro, depth_transform), ssr


def prepare_prompt(
    question: str
    , stage: int
    , n_tor: int
) -> str:
    question = "\n".join([SSRSpecialToken.IMAGE_TOKEN, SSRSpecialToken.DEPTH_TOKEN, question])
    prompt = repeat_special_tokens(
        input_string=construct_conversation(
            question=question
            , rationale=""
            , answer=""
            , stage=stage
            , n_tor=n_tor
        )
        , special_tokens=[SSRSpecialToken.IMAGE_TOKEN, SSRSpecialToken.DEPTH_TOKEN]
        , n_repeats=[(336 // 14) ** 2, (384 // 14) ** 2]
    )
    prompt += "\n<|im_start|>assistant\n"  # Optional
    return prompt


def prepare_vision(
    image_path: str
    , clip_processor: CLIPProcessor
    , siglip_processor: SiglipProcessor
    , depthpro: DepthPro
    , depth_transform: Compose
    , depth_path: str = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    image = Image.open(image_path).convert("RGB")
    image = clip_processor(images=image, return_tensors="pt").pixel_values
    if depth_path is None:
        depth = get_depth(image_path, depthpro, depth_transform).cpu().numpy()
    else:
        depth = np.array(Image.open(change_ext(depth_path, "png")))
    depth = convert_depth(depth, convert_16bits=True, convert_3channels=True)
    depth = siglip_processor(images=depth, return_tensors="pt").pixel_values
    return image, depth


def prepare_data(
    data: Dict[str, Any]
    , clip_processor: CLIPProcessor
    , siglip_processor: SiglipProcessor
    , depthpro: DepthPro
    , depth_transform: Compose
) -> Tuple[str, torch.Tensor, torch.Tensor]:
    prompt = prepare_prompt(
        question=data["question"]
        , stage=3
        , n_tor=10
    )
    image, depth = prepare_vision(
        image_path=data["image_path"]
        , depth_path=data["depth_path"] if "depth_path" in data else None
        , clip_processor=clip_processor
        , siglip_processor=siglip_processor
        , depthpro=depthpro
        , depth_transform=depth_transform
    )
    return prompt, image, depth


def filter_generated_ids(generated_ids: torch.Tensor, tokenizer: Internlm3Tokenizer) -> torch.Tensor:
    start_target = torch.tensor(tokenizer.convert_tokens_to_ids(["<|im_start|>", "assi", "stant", "\n"]), device=generated_ids.device, dtype=generated_ids.dtype)
    end_target = torch.tensor(tokenizer.convert_tokens_to_ids(["<|im_end|>"]), device=generated_ids.device, dtype=generated_ids.dtype)
    start_idx, end_idx = None, None
    for i in range(generated_ids.size(1)):
        if i + start_target.size(0) <= generated_ids.size(1):
            if torch.equal(generated_ids[0, i:i + start_target.size(0)], start_target):
                start_idx = i + start_target.size(0)
        if i + end_target.size(0) <= generated_ids.size(1):
            if torch.equal(generated_ids[0, i:i + end_target.size(0)], end_target):
                end_idx = i
        if start_idx is not None and end_idx is not None:
            break
    if start_idx is not None:
        generated_ids = generated_ids[:, start_idx:]
    if end_idx is not None:
        generated_ids = generated_ids[:, :end_idx]
    return generated_ids


def inference(
    ssr: SSR
    , data: Dict[str, Any]
    , args: argparse.Namespace
    , clip_processor: CLIPProcessor
    , siglip_processor: SiglipProcessor
    , depthpro: DepthPro
    , depth_transform: Compose
    , device: torch.device
) -> str:
    prompt, image, depth = prepare_data(
        data=data
        , clip_processor=clip_processor
        , siglip_processor=siglip_processor
        , depthpro=depthpro
        , depth_transform=depth_transform
    )
    inputs = (ssr.tokenizer(prompt, return_tensors="pt")).to(device)
    with torch.inference_mode():
        generated_ids = ssr.generate(
            input_ids=inputs.input_ids
            , attention_mask=inputs.attention_mask
            , image=image.to(device)
            , depth=depth.to(device)
            , max_new_tokens=64
            , do_sample=args.do_sample
            , num_beams=args.num_beams
            , temperature=args.temperature
            , use_cache=True
        )
    response = ssr.tokenizer.batch_decode(filter_generated_ids(generated_ids[:, inputs.input_ids.size(1):], ssr.tokenizer))[0]
    response = response.strip().strip("\n").strip("</s>")
    return response


def get_data(data_dir: str, task: str) -> List[Dict[str, Any]]:
    with open(os.path.join(data_dir, f"{task}.json"), "r") as f:
        data = json.load(f)
    suffix = ""
    if task == "existence":
        suffix = "Response \"Yes\" or \"No\" only."
    elif task == "counting":
        suffix = "Response the number only"
    data = list(map(
        lambda x: {
            "question": x["question"] + suffix
            , "answer": x["answer"]
            , "image_path": os.path.join(data_dir, x["image"])
            , "depth_path": os.path.join(data_dir, os.sep.join([f"{x['image'].split(os.sep)[0]}_d"] + x["image"].split(os.sep)[1:]))
        }
        , data
    ))
    return data


def calc_metrics(result: List[Dict[str, Any]]) -> Tuple[float, float, float, float]:
    if args.task in ["reach","size"]:
        scores, full_scores = 0, 0
        for j in range(len(result)):
            if j % 2 == 1:
                continue
            pos_response, pos_gt = result[j]["response"], result[j]["answer"]
            neg_response, neg_gt = result[j + 1]["response"], result[j + 1]["answer"]
            accuracy, accuracy_plus = 0, 0
            if pos_gt in pos_response:
                accuracy = accuracy + 1
            if neg_gt in neg_response:
                accuracy = accuracy + 1
            if (pos_gt in pos_response) and (neg_gt in neg_response):
                accuracy_plus = accuracy_plus + 1
            scores = scores + accuracy + accuracy_plus
            full_scores = full_scores + 3
        print(f"{args.task.strip()}: {scores} out of {full_scores}, {100 * scores / full_scores=}%")
    elif args.task.strip() in ["positional","size"]:
        scores, full_scores = 0, 0
        for j in range(len(result)):
            if result[j]["answer"] in result[j]["response"]:
                scores = scores + 1
            full_scores = full_scores + 1
        print(f"{args.task.strip()}: {scores} out of {full_scores}, {100 * scores / full_scores=}%")
    elif args.task.strip() in ["existence"]: 
        scores, full_scores = 0, 0
        for j in range(len(result)):
            if j%2==1: continue
            pos_response, pos_gt = result[j]["response"], result[j]["answer"]
            neg_response, neg_gt = result[j + 1]["response"], result[j + 1]["answer"]
            if (pos_gt in pos_response) and (neg_gt in neg_response):
                scores = scores + 1
            full_scores = full_scores + 1
        print(f"{args.task.strip()}: {scores} out of {full_scores}, {100 * scores / full_scores=}%")
    elif args.task.strip() in ["counting"]:
        errors = []
        for j in range(len(result)):
            error = abs(int(result[j]["response"]) - int(result[j]["answer"])) / int(result[j]["answer"]) * 100
            errors.append(error)
        scores = 100 - sum(errors) / len(errors)
        print(f"{args.task.strip()}: {scores} out of 100.")


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (clip_processor, clip_vision), (siglip_processor, siglip), (depthpro, depth_transform), ssr = load_ptm(
        args.pretrained_path
        , args.clip_path
        , args.siglip_path
        , args.depthpro_path
        , device
    )
    data, result = get_data(args.data_dir, args.task), []
    for item in tqdm(data, desc="Evaluating", ncols=100):
        response = inference(
            ssr=ssr
            , data=item
            , args=args
            , clip_processor=clip_processor
            , siglip_processor=siglip_processor
            , depthpro=depthpro
            , depth_transform=depth_transform
            , device=device
        )
        result.append({
            "response": response
            , "answer": item["answer"]
        })
    # save_dir = os.path.join(os.getcwd(), "result", "spatialbench")
    # os.makedirs(save_dir, exist_ok=True)
    # with open(os.path.join(save_dir, f"{args.task}.json"), "w") as f:
    #     json.dump(result, f, indent=4)
    calc_metrics(result)


if __name__ == "__main__":
    init()
    args = get_args()
    main(args)