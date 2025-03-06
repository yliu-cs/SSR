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
    parser.add_argument("--save_path", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "ssr-cot.jsonl"))
    parser.add_argument("--clip_path", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "clip-vit-large-patch14-336"))
    parser.add_argument("--siglip_path", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "siglip-so400m-patch14-384"))
    parser.add_argument("--max_workers", type=int, default=50)
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


def del_path_prefix(path: str) -> str:
    return os.sep.join(path.split(os.sep)[4:])


def preprocess_llava_cot_100k(
    item: Dict[str, Any]
    , data_dir: str
) -> None:
    data = None
    try:
        question, rationale, answer = extract_question_rationale_answer(item["conversations"])
        image_path = os.path.join(data_dir, item["image"])
        depth_path = change_ext(os.sep.join([data_dir] + [f"{item['image'].split(os.sep)[0]}_d"] + item["image"].split(os.sep)[1:]), "png")
        raw_depth = colorize_depth(depth_path)
        data = {
            "question": question
            , "rationale": rationale
            , "answer": answer
            , "image_path": del_path_prefix(image_path)
            , "depth_path": del_path_prefix(depth_path)
        }
    except:
        pass
    return data


def preprocess_viscot(
    item: Dict[str, Any]
    , data_dir: str
) -> None:
    data = None
    try:
        question, rationale, answer = item["question"], item["rationale"], item["answer"]
        image_path = os.path.join(data_dir, "images", item["image_path"])
        depth_path = change_ext(os.sep.join([data_dir, "images"] + [f"{item['image_path'].split(os.sep)[0]}_d"] + item["image_path"].split(os.sep)[1:]), "png")
        raw_depth = colorize_depth(depth_path)
        data = {
            "question": question
            , "rationale": rationale
            , "answer": answer
            , "image_path": del_path_prefix(image_path)
            , "depth_path": del_path_prefix(depth_path)
        }
    except:
        pass
    return data


def preprocess_vocot(
    item: Dict[str, Any]
    , data_dir: str
) -> None:
    data = None
    try:
        question, rationale, answer = item["question"], item["rationale"], item["answer"]
        image_path = os.path.join(data_dir, "images", item["image_path"])
        depth_path = change_ext(os.sep.join([data_dir, "images"] + [f"{item['image_path'].split(os.sep)[0]}_d"] + item["image_path"].split(os.sep)[1:]), "png")
        raw_depth = colorize_depth(depth_path)
        data = {
            "question": question
            , "rationale": rationale
            , "answer": answer
            , "image_path": del_path_prefix(image_path)
            , "depth_path": del_path_prefix(depth_path)
        }
    except:
        pass
    return data

def preprocess_spatialqa(
    item: Dict[str, Any]
    , data_dir: str
) -> None:
    data = None
    try:
        question, rationale, answer = item["question"], item["rationale"], item["answer"]
        image_path = os.path.join(data_dir, "images", item["image_path"])
        depth_path = change_ext(os.sep.join([data_dir, "images"] + [f"{item['image_path'].split(os.sep)[0]}_d"] + item["image_path"].split(os.sep)[1:]), "png")
        raw_depth = colorize_depth(depth_path)
        data = {
            "question": question
            , "rationale": rationale
            , "answer": answer
            , "image_path": del_path_prefix(image_path)
            , "depth_path": del_path_prefix(depth_path)
        }
    except:
        pass
    return data


def preprocess_data(
    data_dir: str
    , max_workers: int
) -> None:
    dataset, func = None, None
    if "LLaVA-CoT-100k" in data_dir:
        dataset = load_jsonl(os.path.join(data_dir, "train.jsonl"))
        func = preprocess_llava_cot_100k
    elif "Visual-CoT" in data_dir:
        dataset = load_jsonl(os.path.join(data_dir, "ssr_viscot.jsonl"))
        func = preprocess_viscot
    elif "VoCoT" in data_dir:
        dataset = load_jsonl(os.path.join(data_dir, "ssr_vocot.jsonl"))
        func = preprocess_vocot
    elif "SpatialQA" in data_dir:
        dataset = json.load(open((os.path.join(data_dir, "ssr_spatialqa.json")), "r"))
        func = preprocess_spatialqa
    if dataset and func:
        data = thread_map(
            partial(func, data_dir=data_dir)
            , dataset
            , desc=f"Processing {os.path.basename(data_dir)}"
            , max_workers=max_workers
        )
        return data
    else:
        raise ValueError(f"No dataset ({dataset}) or function ({func}) to process {data_dir}")


def main(args: Namespace) -> None:
    data = []
    for data_dir in args.data_dirs:
        data += list(filter(lambda x: x is not None, preprocess_data(data_dir, args.max_workers)))
    with open(args.save_path, "w", encoding="utf-8") as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    quiet()
    args = get_args()
    main(args)