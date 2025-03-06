import os
import re
import json
import torch
import autoroot
import numpy as np
from PIL import Image
from functools import partial
from typing import List, Tuple, Any, Dict
from argparse import Namespace, ArgumentParser
from tqdm.contrib.concurrent import thread_map
from transformers import CLIPProcessor, CLIPVisionModel, SiglipProcessor, SiglipVisionModel
from ssr.utils.misc import quiet, load_jsonl, change_ext, colorize_depth, hash_str, get_chunk


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--data_dirs", type=List[str], default=[
        os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "LLaVA-CoT-100k")
        , os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "Visual-CoT")
        , os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "VoCoT")
        , os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "SpatialQA")
    ])
    parser.add_argument("--save_dir", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "SSR-CoT"))
    parser.add_argument("--clip_path", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "clip-vit-large-patch14-336"))
    parser.add_argument("--siglip_path", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "siglip-so400m-patch14-384"))
    parser.add_argument("--max_workers", type=int, default=50)
    parser.add_argument("--num_chunks", type=int, default=4)
    parser.add_argument("--chunk_idx", type=int, default=0)
    return parser.parse_args()


def parse_special_tokens(text: str) -> Tuple[str, str]:
    answer_match = re.search(r"<CONCLUSION>(.*?)</CONCLUSION>", text, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else None
    text_without_answer = re.sub(r"<CONCLUSION>.*?</CONCLUSION>", "", text, flags=re.DOTALL)
    rationale = re.sub(r"<\/?[A-Z]+>", "", text_without_answer)
    rationale = re.sub(r"\s+", " ", rationale).strip()
    if answer:
        answer = re.sub(r"\s+", " ", answer).strip()
    return rationale, answer


def extract_question_rationale_answer(conversations: List[Dict[str, Any]]) -> Tuple[str, str, str]:
    question, rationale, answer = None, None, None
    for conv in conversations:
        if conv["from"] == "human":
            question = conv["value"]
        else:
            rationale, answer = parse_special_tokens(conv["value"])
        if question and rationale and answer:
            break
    return question, rationale, answer


def get_visual_embeds(
    raw_image: np.ndarray
    , raw_depth: np.ndarray
    , clip_processor: CLIPProcessor
    , clip_model: CLIPVisionModel
    , siglip_processor: SiglipProcessor
    , siglip_model: SiglipVisionModel
) -> Tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        image_embeds = (clip_model(**(clip_processor(images=raw_image, return_tensors="pt").to("cuda"))).last_hidden_state).squeeze(0).detach().cpu().numpy()
        depth_embeds = (siglip_model(**(siglip_processor(images=raw_depth, return_tensors="pt").to("cuda"))).last_hidden_state).squeeze(0).detach().cpu().numpy()
    return image_embeds, depth_embeds


def preprocess_llava_cot_100k(
    item: Dict[str, Any]
    , data_dir: str
    , save_dir: str
    , clip_processor: CLIPProcessor
    , clip_model: CLIPVisionModel
    , siglip_processor: SiglipProcessor
    , siglip_model: SiglipVisionModel
) -> None:
    try:
        question, rationale, answer = extract_question_rationale_answer(item["conversations"])
        image_path = os.path.join(data_dir, item["image"])
        raw_image = np.array(Image.open(image_path).convert("RGB"))
        depth_path = change_ext(os.sep.join([data_dir] + [f"{item['image'].split(os.sep)[0]}_d"] + item["image"].split(os.sep)[1:]), "png")
        raw_depth = colorize_depth(depth_path)
        image_embeds, depth_embeds = get_visual_embeds(raw_image, raw_depth, clip_processor, clip_model, siglip_processor, siglip_model)
        data = {
            "question": question
            , "rationale": rationale
            , "answer": answer
            , "image_path": image_path
            , "image_embeds": image_embeds
            , "depth_path": depth_path
            , "depth_embeds": depth_embeds
        }
        np.save(os.path.join(save_dir, f"{hash_str(data_dir + question + rationale + answer)}.npy"), data)
    except:
        pass


def preprocess_viscot(
    item: Dict[str, Any]
    , data_dir: str
    , save_dir: str
    , clip_processor: CLIPProcessor
    , clip_model: CLIPVisionModel
    , siglip_processor: SiglipProcessor
    , siglip_model: SiglipVisionModel
) -> None:
    try:
        question, rationale, answer = item["question"], item["rationale"], item["answer"]
        image_path = os.path.join(data_dir, "images", item["image_path"])
        raw_image = np.array(Image.open(image_path).convert("RGB"))
        depth_path = change_ext(os.sep.join([data_dir, "images"] + [f"{item['image_path'].split(os.sep)[0]}_d"] + item["image_path"].split(os.sep)[1:]), "png")
        raw_depth = colorize_depth(depth_path)
        image_embeds, depth_embeds = get_visual_embeds(raw_image, raw_depth, clip_processor, clip_model, siglip_processor, siglip_model)
        data = {
            "question": question
            , "rationale": rationale
            , "answer": answer
            , "image_path": image_path
            , "image_embeds": image_embeds
            , "depth_path": depth_path
            , "depth_embeds": depth_embeds
        }
        np.save(os.path.join(save_dir, f"{hash_str(data_dir + question + rationale + answer)}.npy"), data)
    except:
        pass


def preprocess_vocot(
    item: Dict[str, Any]
    , data_dir: str
    , save_dir: str
    , clip_processor: CLIPProcessor
    , clip_model: CLIPVisionModel
    , siglip_processor: SiglipProcessor
    , siglip_model: SiglipVisionModel
) -> None:
    try:
        question, rationale, answer = item["question"], item["rationale"], item["answer"]
        image_path = os.path.join(data_dir, "images", item["image_path"])
        raw_image = np.array(Image.open(image_path).convert("RGB"))
        depth_path = change_ext(os.sep.join([data_dir, "images"] + [f"{item['image_path'].split(os.sep)[0]}_d"] + item["image_path"].split(os.sep)[1:]), "png")
        raw_depth = colorize_depth(depth_path)
        image_embeds, depth_embeds = get_visual_embeds(raw_image, raw_depth, clip_processor, clip_model, siglip_processor, siglip_model)
        data = {
            "question": question
            , "rationale": rationale
            , "answer": answer
            , "image_path": image_path
            , "image_embeds": image_embeds
            , "depth_path": depth_path
            , "depth_embeds": depth_embeds
        }
        np.save(os.path.join(save_dir, f"{hash_str(data_dir + question + rationale + answer)}.npy"), data)
    except:
        pass


def preprocess_spatialqa(
    item: Dict[str, Any]
    , data_dir: str
    , save_dir: str
    , clip_processor: CLIPProcessor
    , clip_model: CLIPVisionModel
    , siglip_processor: SiglipProcessor
    , siglip_model: SiglipVisionModel
) -> None:
    try:
        question, rationale, answer = item["question"], item["rationale"], item["answer"]
        image_path = os.path.join(data_dir, "images", item["image_path"])
        raw_image = np.array(Image.open(image_path).convert("RGB"))
        depth_path = change_ext(os.sep.join([data_dir, "images"] + [f"{item['image_path'].split(os.sep)[0]}_d"] + item["image_path"].split(os.sep)[1:]), "png")
        raw_depth = colorize_depth(depth_path)
        image_embeds, depth_embeds = get_visual_embeds(raw_image, raw_depth, clip_processor, clip_model, siglip_processor, siglip_model)
        data = {
            "question": question
            , "rationale": rationale
            , "answer": answer
            , "image_path": image_path
            , "image_embeds": image_embeds
            , "depth_path": depth_path
            , "depth_embeds": depth_embeds
        }
        np.save(os.path.join(save_dir, f"{hash_str(data_dir + question + rationale + answer)}.npy"), data)
    except:
        pass


def preprocess_data(
    data_dir: str
    , save_dir: str
    , clip_processor: CLIPProcessor
    , clip_model: CLIPVisionModel
    , siglip_processor: SiglipProcessor
    , siglip_model: SiglipVisionModel
    , num_chunks: int
    , chunk_idx: int
    , max_workers: int
) -> None:
    dataset, func = None, None
    if "LLaVA-CoT-100k" in data_dir:
        dataset = get_chunk(load_jsonl(os.path.join(data_dir, "train.jsonl")), n=num_chunks, k=chunk_idx)
        func = preprocess_llava_cot_100k
    elif "Visual-CoT" in data_dir:
        dataset = get_chunk(load_jsonl(os.path.join(data_dir, "ssr_viscot.jsonl")), n=num_chunks, k=chunk_idx)
        func = preprocess_viscot
    elif "VoCoT" in data_dir:
        dataset = get_chunk(load_jsonl(os.path.join(data_dir, "ssr_vocot.jsonl")), n=num_chunks, k=chunk_idx)
        func = preprocess_vocot
    elif "SpatialQA" in data_dir:
        dataset = get_chunk(json.load(open((os.path.join(data_dir, "ssr_spatialqa.json")), "r")), n=num_chunks, k=chunk_idx)
        func = preprocess_spatialqa
    if dataset and func:
        thread_map(
            partial(
                func
                , data_dir=data_dir
                , save_dir=save_dir
                , clip_processor=clip_processor
                , clip_model=clip_model
                , siglip_processor=siglip_processor
                , siglip_model=siglip_model
            )
            , dataset
            , desc=f"[{chunk_idx}/{num_chunks}] Processing {os.path.basename(data_dir)}"
            , max_workers=max_workers
        )
    else:
        raise ValueError(f"No dataset ({dataset}) or function ({func}) to process {data_dir}")


def main(args: Namespace) -> None:
    os.makedirs(args.save_dir, exist_ok=True)
    clip_processor, clip_model = CLIPProcessor.from_pretrained(args.clip_path), CLIPVisionModel.from_pretrained(args.clip_path).to("cuda")
    siglip_processor, siglip_model = SiglipProcessor.from_pretrained(args.siglip_path), SiglipVisionModel.from_pretrained(args.siglip_path).to("cuda")
    for data_dir in args.data_dirs:
        preprocess_data(
            data_dir
            , args.save_dir
            , clip_processor
            , clip_model
            , siglip_processor
            , siglip_model
            , args.num_chunks
            , args.chunk_idx
            , args.max_workers
        )


if __name__ == "__main__":
    quiet()
    args = get_args()
    main(args)