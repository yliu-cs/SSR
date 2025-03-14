import os
import json
import torch
import autoroot
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from ssr.models.midi import MIDI
from ssr.models.vlm import SSRVLM
from qwen_vl_utils import process_vision_info
from argparse import Namespace, ArgumentParser
from typing import List, Dict, Union, Tuple, Any
from ssr.utils.prompt import SSRSpecialToken, insert_tor
from ssr.utils.misc import quiet, str_datetime, change_ext, colorize_depth
from transformers import AutoTokenizer, Qwen2_5_VLProcessor, CLIPProcessor, CLIPVisionModel, SiglipProcessor, SiglipVisionModel, PreTrainedTokenizer


TASK = {
    "SpatialBench": ["positional", "existence", "counting", "reach", "size"]
    , "MME": ["OCR", "celebrity", "color", "count", "landmark", "position", "scene", "artwork", "code_reasoning", "commonsense_reasoning", "existence", "numerical_calculation", "posters", "text_translation"]
}


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Dataset"))
    parser.add_argument("--mamba", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "mamba-130m-hf"))
    parser.add_argument("--clip_path", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "clip-vit-large-patch14-336"))
    parser.add_argument("--siglip_path", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "siglip-so400m-patch14-384"))
    parser.add_argument("--pretrained_midi", type=str, default=os.path.join(os.getcwd(), "checkpoints", "SSR-Reasoning", "2025-03-08 23:35:10"))
    parser.add_argument("--pretrained_vlm", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "Qwen2.5-VL-3B-Instruct"))
    parser.add_argument("--n_tor", type=int, default=10)
    parser.add_argument("--image_size", type=Tuple[int, int], default=(256, 256))
    parser.add_argument("--benchmark", type=str, default="SpatialBench", choices=["SpatialBench", "MME"])
    return parser.parse_args()


def get_visual_embeds(
    raw_image: np.ndarray
    , raw_depth: np.ndarray
    , clip_processor: CLIPProcessor
    , clip_model: CLIPVisionModel
    , siglip_processor: SiglipProcessor
    , siglip_model: SiglipVisionModel
) -> Tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        image_embeds = (clip_model(**(clip_processor(images=raw_image, return_tensors="pt").to(clip_model.device))).last_hidden_state).detach()
        depth_embeds = (siglip_model(**(siglip_processor(images=raw_depth, return_tensors="pt").to(siglip_model.device))).last_hidden_state).detach()
    return image_embeds, depth_embeds


def load_spatialbench_data(data_dir: str) -> List[str]:
    data = {}
    for task in TASK["SpatialBench"]:
        with open(os.path.join(data_dir, f"{task}.json"), "r") as f:
            data[task] = json.load(f)
        suffix = ""
        if task == "existence":
            suffix = "Response \"Yes\" or \"No\" only."
        elif task == "counting":
            suffix = "Response the number only"
        data[task] = list(map(
            lambda x: {
                "question": x["question"] + suffix
                , "answer": x["answer"]
                , "image_path": os.path.join(data_dir, x["image"])
                , "depth_path": change_ext(os.path.join(data_dir, os.sep.join([f"{x['image'].split(os.sep)[0]}_d"] + x["image"].split(os.sep)[1:])), "png")
            }
            , data[task]
        ))
    return data


def load_mme_data(data_dir: str) -> List[str]:
    data = {}
    for task in TASK["MME"]:
        data[task] = []
        with open(os.path.join(data_dir, "QA", f"{task}.txt"), 'r', encoding='utf-8') as f:
            for line in f:
                image_path, question, answer = line.strip("\n").split("\t")
                if task in ["celebrity", "artwork", "posters", "scene", "landmark"]:
                    image_path = os.path.join(data_dir, "MME_Benchmark", task, "images", image_path)
                else:
                    image_path = os.path.join(data_dir, "MME_Benchmark", task, image_path.split(os.sep)[-1])
                data[task].append({
                    "question": question
                    , "answer": answer
                    , "image_path": image_path
                })
    return data


def load_data(data_dir: str, benchmark: str) -> List[str]:
    func_map = {
        "SpatialBench": load_spatialbench_data
        , "MME": load_mme_data
    }
    assert benchmark in func_map
    return func_map[benchmark](data_dir)


def prepare_data(
    sample: Dict[str, str]
    , mamba_tokenizer: PreTrainedTokenizer
    , clip_processor: CLIPProcessor
    , clip_model: CLIPVisionModel
    , siglip_processor: SiglipProcessor
    , siglip_model: SiglipVisionModel
    , vlm_processor: Qwen2_5_VLProcessor
    , device: torch.device
    , n_tor: int = 10
    , image_size: Tuple[int, int] = (256, 256)
) -> Dict[str, torch.Tensor]:
    question, image_path, depth_path = sample["question"], sample["image_path"], sample["depth_path"]
    mamba_question = mamba_tokenizer(question + insert_tor("", n_tor), add_special_tokens=False, return_tensors="pt")
    mamba_input_ids = mamba_question.input_ids
    mamba_attention_mask = mamba_question.attention_mask
    raw_image = Image.open(image_path).convert("RGB")
    raw_depth = colorize_depth(depth_path)
    image_embeds, depth_embeds = get_visual_embeds(np.array(raw_image), raw_depth, clip_processor, clip_model, siglip_processor, siglip_model)
    mamba_attention_mask = torch.cat((torch.ones(image_embeds.size(1) + depth_embeds.size(1), dtype=mamba_attention_mask.dtype).unsqueeze(0), mamba_attention_mask), dim=1)
    messages = [{
        "role": "user"
        , "content": [
            {"type": "image", "image": raw_image.resize(image_size)}
            , {"type": "text", "text": insert_tor("", n_tor) + question}
        ]
    }]
    text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    vlm_inputs = vlm_processor(text=[text], images=image_inputs, videos=None, padding=True, return_tensors="pt")
    vlm_input_ids = vlm_inputs.input_ids
    vlm_attention_mask = vlm_inputs.attention_mask
    vlm_pixel_values = vlm_inputs.pixel_values
    vlm_image_grid_thw = vlm_inputs.image_grid_thw
    # print(f"{mamba_input_ids.size()=} {mamba_attention_mask.size()=}")
    # print(f"{image_embeds.size()=} {depth_embeds.size()=}")
    # print(f"{vlm_input_ids.size()=} {vlm_attention_mask.size()=} {vlm_pixel_values.size()=} {vlm_image_grid_thw.size()=}")
    return {
        "mamba_input_ids": mamba_input_ids.to(device)
        , "mamba_attention_mask": mamba_attention_mask.to(device)
        , "image_embeds": image_embeds.to(device)
        , "depth_embeds": depth_embeds.to(device)
        , "vlm_input_ids": vlm_input_ids.to(device)
        , "vlm_attention_mask": vlm_attention_mask.to(device)
        , "vlm_pixel_values": vlm_pixel_values.to(device)
        , "vlm_image_grid_thw": vlm_image_grid_thw.to(device)
    }


def inference(
    sample: Dict[str, str]
    , midi: MIDI
    , vlm: SSRVLM
    , mamba_tokenizer: PreTrainedTokenizer
    , vlm_processor: Qwen2_5_VLProcessor
    , clip_processor: CLIPProcessor
    , clip_model: CLIPVisionModel
    , siglip_processor: SiglipProcessor
    , siglip_model: SiglipVisionModel
    , n_tor: int
    , tor_token_id: Tuple[int, int]
    , image_size: Tuple[int, int]
    , device: torch.device
) -> None:
    sample = prepare_data(
        sample=sample
        , mamba_tokenizer=mamba_tokenizer
        , clip_processor=clip_processor
        , clip_model=clip_model
        , siglip_processor=siglip_processor
        , siglip_model=siglip_model
        , vlm_processor=vlm_processor
        , device=device
        , n_tor=n_tor
        , image_size=image_size
    )
    with torch.no_grad():
        tor_embeds = midi(
            mamba_input_ids=sample["mamba_input_ids"]
            , mamba_attention_mask=sample["mamba_attention_mask"]
            , image_embeds=sample["image_embeds"]
            , depth_embeds=sample["depth_embeds"]
            , tor_token_id=tor_token_id
            , alignment=False
        ).tor_embeds
    with torch.inference_mode():
        generated_ids = vlm.generate(
            input_ids=sample["vlm_input_ids"]
            , attention_mask=sample["vlm_attention_mask"]
            , pixel_values=sample["vlm_pixel_values"]
            , image_grid_thw=sample["vlm_image_grid_thw"]
            , max_new_tokens=128
            , tor_embeds=tor_embeds
            , tor_token_id=tor_token_id[1]
        )
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(sample["vlm_input_ids"], generated_ids)]
    output_text = vlm_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_text[0]


def evaluate(
    data: Union[List[str], Dict[str, List[str]]]
    , midi: MIDI
    , vlm: SSRVLM
    , mamba_tokenizer: PreTrainedTokenizer
    , vlm_processor: Qwen2_5_VLProcessor
    , clip_processor: CLIPProcessor
    , clip_model: CLIPVisionModel
    , siglip_processor: SiglipProcessor
    , siglip_model: SiglipVisionModel
    , n_tor: int
    , tor_token_id: Tuple[int, int]
    , image_size: Tuple[int, int]
    , device: torch.device
) -> None:
    if isinstance(data, dict):
        result = {}
        for task in data:
            result[task] = []
            for sample in tqdm(data[task], desc=f"Evaluating SpatialBench {task}"):
                answer = sample["answer"]
                response = inference(
                    sample
                    , midi, vlm
                    , mamba_tokenizer, vlm_processor
                    , clip_processor, clip_model
                    , siglip_processor, siglip_model
                    , n_tor, tor_token_id
                    , image_size
                    , device
                )
                result[task].append({
                    "response": response
                    , "answer": answer
                })
    else:
        result = []
        for sample in tqdm(data, desc="Evaluating MME"):
            response = inference(
                sample
                , midi, vlm
                , mamba_tokenizer, vlm_processor
                , clip_processor, clip_model
                , siglip_processor, siglip_model
                , n_tor, tor_token_id
                , image_size
            )
            result.append({
                "response": response
                , "answer": sample["answer"]
            })
    return result


def calc_spatialbench_metrics(full_result: Dict[str, List[Dict[str, Any]]]) -> None:
    metrics = {}
    for task in TASK["SpatialBench"]:
        result = full_result[task]
        if task in ["reach","size"]:
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
            metrics[task] = 100 * scores / full_scores
        elif task.strip() in ["positional","size"]:
            scores, full_scores = 0, 0
            for j in range(len(result)):
                if result[j]["answer"] in result[j]["response"]:
                    scores = scores + 1
                full_scores = full_scores + 1
            metrics[task] = 100 * scores / full_scores
        elif task.strip() in ["existence"]: 
            scores, full_scores = 0, 0
            for j in range(len(result)):
                if j%2==1: continue
                pos_response, pos_gt = result[j]["response"], result[j]["answer"]
                neg_response, neg_gt = result[j + 1]["response"], result[j + 1]["answer"]
                if (pos_gt in pos_response) and (neg_gt in neg_response):
                    scores = scores + 1
                full_scores = full_scores + 1
            metrics[task] = 100 * scores / full_scores
        elif task.strip() in ["counting"]:
            errors = []
            for j in range(len(result)):
                error = abs(int(result[j]["response"]) - int(result[j]["answer"])) / int(result[j]["answer"]) * 100
                errors.append(error)
            scores = 100 - sum(errors) / len(errors)
            metrics[task] = scores
    return metrics


def main(args: Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{str_datetime()} Loading Tokenizer & Processor...")
    mamba_tokenizer = AutoTokenizer.from_pretrained(args.mamba)
    mamba_tokenizer.add_tokens(SSRSpecialToken.TOR_TOKEN, special_tokens=True)
    vlm_processor = Qwen2_5_VLProcessor.from_pretrained(args.pretrained_vlm)
    vlm_processor.tokenizer.add_tokens(SSRSpecialToken.TOR_TOKEN, special_tokens=True)
    print(f"{str_datetime()} Loading CLIP and Siglip Models...")
    clip_processor, clip_model = CLIPProcessor.from_pretrained(args.clip_path), (CLIPVisionModel.from_pretrained(args.clip_path)).to(device)
    siglip_processor, siglip_model = SiglipProcessor.from_pretrained(args.siglip_path), (SiglipVisionModel.from_pretrained(args.siglip_path)).to(device)

    print(f"{str_datetime()} Loading MIDI Model...")
    midi = MIDI.from_pretrained(args.pretrained_midi, device_map=device)
    midi.eval()
    print(f"{str_datetime()} Loading SSRVLM Model...")
    vlm = SSRVLM.from_pretrained(args.pretrained_vlm, device_map=device)
    vlm.eval()

    print(f"{str_datetime()} Loading Data...")
    data = load_data(os.path.join(args.data_dir, args.benchmark), args.benchmark)
    tor_token_id = (mamba_tokenizer._tokenizer.token_to_id(SSRSpecialToken.TOR_TOKEN), vlm_processor.tokenizer._tokenizer.token_to_id(SSRSpecialToken.TOR_TOKEN))
    print(f"{str_datetime()} Evaluating...")
    result = evaluate(
        data
        , midi, vlm
        , mamba_tokenizer, vlm_processor
        , clip_processor, clip_model
        , siglip_processor, siglip_model
        , args.n_tor, tor_token_id
        , args.image_size
        , device
    )
    print(f"{str_datetime()} Calculating Metrics...")
    if args.benchmark == "SpatialBench":
        metrics = calc_spatialbench_metrics(result)
        for task, score in metrics.items():
            print(f"{str_datetime()} {task:<10}: {round(score, 1)}")
    # elif args.benchmark == "MME":
    #     calc_mme_metrics(result)


if __name__ == "__main__":
    quiet()
    args = get_args()
    main(args)