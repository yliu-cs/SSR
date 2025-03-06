import os
import json
import torch
import autoroot
import numpy as np
from PIL import Image
from tqdm import tqdm
from random import choices
from ast import literal_eval
from typing import List, Dict, Any
from qwen_vl_utils import process_vision_info
from argparse import Namespace, ArgumentParser
from ssr.utils.prompt import string_truncation
from ssr.utils.misc import quiet, freeze_module, str_datetime, get_chunk
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, Qwen2_5_VLForConditionalGeneration


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "SSR-CoT"))
    parser.add_argument("--vlm_path", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "Qwen2.5-VL-7B-Instruct"))
    parser.add_argument("--llm_path", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "Qwen2.5-14B-Instruct-1M"))
    parser.add_argument("--num_chunks", type=int, default=10)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--n", type=int, default=3000)
    return parser.parse_args()


def load_evaluate_data(data_dir: str, n: int) -> List[Dict[str, Any]]:
    # return list(map(
    #     lambda x: np.load(os.path.join(data_dir, x), allow_pickle=True).item()
    #     , choices(os.listdir(data_dir), k=n)
    # ))
    ret = []
    for file in choices(os.listdir(data_dir), k=n):
        try:
            data = np.load(os.path.join(data_dir, file), allow_pickle=True).item()
            ret.append(data)
        except Exception as e:
            print(f"{str_datetime()}: {e=}")
            continue
    return ret


def inference(
    image: np.ndarray
    , question: str
    , processor: AutoProcessor
    , model: Qwen2_5_VLForConditionalGeneration
) -> str:
    image = Image.fromarray(image)
    image.resize((256, 256))
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
) -> float:
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
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return literal_eval(response)["score"]


def eval_item(
    item: Dict[str, Any]
    , processor: AutoProcessor
    , vlm: Qwen2_5_VLForConditionalGeneration
    , llm: AutoModelForCausalLM
    , tokenizer: AutoTokenizer
) -> None:
    question, rationale, answer = item["question"], item["rationale"], item["answer"]
    question, rationale, answer = (string_truncation(text, processor.tokenizer, max_len) for text, max_len in zip((question, rationale, answer), (256, 1024, 256)))
    image = item["image"]
    qa_response, qra_response = None, None
    try:
        qa_response = inference(image, question, processor, vlm)
    except Exception as e:
        print(f"{len(question)=}")
        print(f"{e=}")
    # print(f"{qa_response=}")
    try:
        qra_response = inference(image, f"{rationale}\nPlease answer the following questions following the above basic reasoning content.\n{question}", processor, vlm)
    except Exception as e:
        print(f"{len(rationale)=}")
        print(f"{e=}")
    # print(f"{qra_response=}")
    if qa_response is None or qra_response is None:
        return 0, 0
    try:
        qa_score = get_score(question, qa_response, answer, llm, tokenizer)
        qra_score = get_score(question, qra_response, answer, llm, tokenizer)
    except Exception as e:
        print(f"{e=}")
        qa_score, qra_score = 0, 0
    return qa_score, qra_score


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
    data = load_evaluate_data(args.data_dir, args.n)
    data = get_chunk(data, args.num_chunks, args.chunk_idx)

    qa_scores, qra_scores = [], []
    for item in tqdm(data, desc=f"[{args.chunk_idx}|{args.num_chunks}] Evaluating Explicit CoT", ncols=100):
        qa_score, qra_score = eval_item(item, processor, vlm, llm, tokenizer)
        qa_scores.append(qa_score)
        qra_scores.append(qra_score)
    
    save_dir = os.path.join(os.getcwd(), "media", "exp_cot_eval")
    print(f"{str_datetime()}: Saving results to {save_dir}...")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, f"qa_scores_{args.chunk_idx}.npy"), qa_scores)
    np.save(os.path.join(save_dir, f"qra_scores_{args.chunk_idx}.npy"), qra_scores)


if __name__ == "__main__":
    quiet()
    args = get_args()
    main(args)