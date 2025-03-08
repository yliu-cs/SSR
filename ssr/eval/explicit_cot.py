import os
import json
import torch
import autoroot
import numpy as np
from PIL import Image
from tqdm import tqdm
from random import choices
from ast import literal_eval
from typing import List, Dict, Any, Tuple
from qwen_vl_utils import process_vision_info
from argparse import Namespace, ArgumentParser
from ssr.utils.prompt import string_truncation
from ssr.utils.misc import quiet, freeze_module, str_datetime, get_chunk, load_jsonl
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, Qwen2_5_VLForConditionalGeneration


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Dataset"))
    parser.add_argument("--vlm_path", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "Qwen2.5-VL-7B-Instruct"))
    parser.add_argument("--llm_path", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "Qwen2.5-14B-Instruct-1M"))
    parser.add_argument("--num_chunks", type=int, default=10)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--n", type=int, default=10000)
    return parser.parse_args()


def inference(
    image: np.ndarray
    , question: str
    , processor: AutoProcessor
    , model: Qwen2_5_VLForConditionalGeneration
) -> str:
    messages = [{
        "role": "user"
        , "content": [
            {"type": "image", "image": image}
            , {"type": "text", "text": question}
        ]
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text]
        , images=image_inputs
        , videos=video_inputs
        , padding=True
        , return_tensors="pt"
    )
    inputs = inputs.to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_text


def get_score(
    question: str
    , response: str
    , answer: str
    , llm: AutoModelForCausalLM
    , tokenizer: AutoTokenizer
) -> Tuple[str, float]:
    messages = [
        {
            "role": "system"
            , "content":
                "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs."
                "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                "------"
                "##INSTRUCTIONS: "
                "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                "- Consider synonyms or paraphrases as valid matches.\n"
                "- Evaluate the correctness of the prediction compared to the answer."
        }
        , {
            "role": "user",
            "content":
                "Please evaluate the following image-based question-answer pair:\n\n"
                f"Question: {question}\n"
                f"Correct Answer: {answer}\n"
                f"Predicted Answer: {response}\n\n"
                "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
        }
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    llm_inputs = tokenizer([text], return_tensors="pt").to(llm.device)
    generated_ids = llm.generate(**llm_inputs, max_new_tokens=256)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(llm_inputs.input_ids, generated_ids)]
    response = literal_eval(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
    return response["pred"], response["score"]


def eval_item(
    item: Dict[str, Any]
    , processor: AutoProcessor
    , vlm: Qwen2_5_VLForConditionalGeneration
    , llm: AutoModelForCausalLM
    , tokenizer: AutoTokenizer
    , data_dir: str
) -> Tuple[Tuple[str, float], Tuple[str, float]]:
    question, rationale, answer = item["question"], item["rationale"], item["answer"]
    question, rationale, answer = (string_truncation(text, processor.tokenizer, max_len) for text, max_len in zip((question, rationale, answer), (256, 1024, 256)))
    image = Image.open(os.path.join(data_dir, item["image_path"])).convert("RGB")
    qa_response = inference(image, question, processor, vlm)
    qra_response = inference(image, f"{rationale}\nPlease answer the following questions following the above basic reasoning content.\n{question}", processor, vlm)
    qa_pred, qa_score = get_score(question, qa_response, answer, llm, tokenizer)
    qra_pred, qra_score = get_score(question, qra_response, answer, llm, tokenizer)
    qa_pred, qra_pred = qa_pred.strip().lower(), qra_pred.strip().lower()
    if qa_pred not in ["yes", "no"] or qra_pred not in ["yes", "no"]:
        raise ValueError
    return (qa_pred, qa_score), (qra_pred, qra_score)


def main(args: Namespace) -> None:
    print(f"{str_datetime()}: Loading VLM from {args.vlm_path}...")
    processor = AutoProcessor.from_pretrained(args.vlm_path)
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.vlm_path, torch_dtype=torch.bfloat16, device_map="cuda")
    freeze_module(vlm)
    vlm.eval()

    print(f"{str_datetime()}: Loading LLM from {args.llm_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
    llm = AutoModelForCausalLM.from_pretrained(args.llm_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="cuda")
    freeze_module(llm)
    llm.eval()

    print(f"{str_datetime()}: Loading data from {args.data_dir}...")
    data = choices(load_jsonl(os.path.join(args.data_dir, "ssr-cot.jsonl")), k=args.n)
    data = get_chunk(data, args.num_chunks, args.chunk_idx)

    qa_preds, qa_scores, qra_preds, qra_scores = [], [], [], []
    for item in tqdm(data, desc=f"[{args.chunk_idx}|{args.num_chunks}] Evaluating Explicit CoT", ncols=100):
        try:
            (qa_pred, qa_score), (qra_pred, qra_score) = eval_item(item, processor, vlm, llm, tokenizer, args.data_dir)
        except Exception as e:
            print(f"{str_datetime()}: {e=}")
            continue
        qa_preds.append(qa_pred)
        qa_scores.append(qa_score)
        qra_preds.append(qra_pred)
        qra_scores.append(qra_score)
    qa_preds = list(map(lambda x: 1 if x == "yes" else 0, qa_preds))
    qra_preds = list(map(lambda x: 1 if x == "yes" else 0, qra_preds))
    
    save_dir = os.path.join(os.getcwd(), "media", "exp_cot_eval")
    print(f"{str_datetime()}: Saving results to {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"qa_preds_{args.chunk_idx}.npy"), qa_preds)
    np.save(os.path.join(save_dir, f"qa_scores_{args.chunk_idx}.npy"), qa_scores)
    np.save(os.path.join(save_dir, f"qra_preds_{args.chunk_idx}.npy"), qra_preds)
    np.save(os.path.join(save_dir, f"qra_scores_{args.chunk_idx}.npy"), qra_scores)


if __name__ == "__main__":
    quiet()
    args = get_args()
    main(args)