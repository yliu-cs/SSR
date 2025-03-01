import os
import torch
import argparse
import autoroot
import depth_pro
from PIL import Image
from tqdm import tqdm
from depth_pro.depth_pro import DepthPro
from typing import List, Dict, Any, Tuple
from torchvision.transforms import Compose
from ssr.models.modeling_ssr import SSRConfig, SSR
from transformers import CLIPProcessor, SiglipProcessor
from ssr.utils.misc import init, str_datetime, convert_depth
from ssr.models.tokenization_internlm3 import Internlm3Tokenizer
from ssr.utils.load_ptm import load_clip_vit, load_siglip, load_depth_pro
from ssr.utils.prompt import SSRSpecialToken, construct_conversation, repeat_special_tokens


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Checkpoint", "SSR", "stage2"))
    parser.add_argument("--clip_path", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "clip-vit-large-patch14-336"))
    parser.add_argument("--siglip_path", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "siglip-so400m-patch14-384"))
    parser.add_argument("--depthpro_path", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "DepthPro"))
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "MME"))
    parser.add_argument("--save_dir", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "MME", "Your_Results"))
    parser.add_argument("--task", type=str, default="OCR", choices=["OCR", "celebrity", "color", "count", "landmark", "position", "scene", "artwork", "code_reasoning", "commonsense_reasoning", "existence", "numerical_calculation", "posters", "text_translation"])
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    image = Image.open(image_path).convert("RGB")
    image = clip_processor(images=image, return_tensors="pt").pixel_values
    depth = get_depth(image_path, depthpro, depth_transform)
    depth = convert_depth(depth.cpu().numpy(), convert_16bits=True, convert_3channels=True)
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
            , do_sample=False
            , num_beams=1
            , max_new_tokens=256
            , top_p=None
            , temperature=0.
            , use_cache=True
        )
    response = ssr.tokenizer.batch_decode(filter_generated_ids(generated_ids[:, inputs.input_ids.size(1):], ssr.tokenizer))[0]
    return response


def get_data(data_dir: str, task: str) -> List[Dict[str, Any]]:
    data = []
    with open(os.path.join(data_dir, "QA", f"{task}.txt"), 'r', encoding='utf-8') as f:
        for line in f:
            image_path, question, answer = line.strip("\n").split("\t")
            if task in ["celebrity", "artwork", "posters", "scene", "landmark"]:
                image_path = os.path.join(data_dir, "MME_Benchmark", task, "images", image_path)
            else:
                image_path = os.path.join(data_dir, "MME_Benchmark", task, image_path.split(os.sep)[-1])
            data.append({
                "question": question
                , "answer": answer
                , "image_path": image_path
            })
    return data


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(os.path.join(args.save_dir, f"{args.task}.txt")):
        return
    (clip_processor, clip_vision), (siglip_processor, siglip), (depthpro, depth_transform), ssr = load_ptm(
        args.pretrained_path
        , args.clip_path
        , args.siglip_path
        , args.depthpro_path
        , device
    )
    data, result = get_data(args.data_dir, args.task), []
    for item in tqdm(data, desc=f"Evaluating {args.task}", ncols=100):
        try:
            response = inference(
                ssr=ssr
                , data=item
                , clip_processor=clip_processor
                , siglip_processor=siglip_processor
                , depthpro=depthpro
                , depth_transform=depth_transform
                , device=device
            )
            result.append({
                "image_path": item["image_path"]
                , "question": item["question"]
                , "answer": item["answer"]
                , "response": response
            })
        except Exception as e:
            print(f"Error: {e}")
            continue
    with open(os.path.join(args.save_dir, f"{args.task}.txt"), "w") as f:
        for item in result:
            image_path = item["image_path"].split(os.sep)[-1]
            question = item["question"].replace("\n", " ")  
            answer = item["answer"].replace("\n", " ")
            response = item["response"].replace("\t", "").replace("\n", "")
            f.write(f"{image_path}\t{question}\t{answer}\t{response}\n")


if __name__ == "__main__":
    init()
    args = get_args()
    main(args)