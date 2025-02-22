import os
import json
import base64
import random
import autoroot
import numpy as np
from tqdm import tqdm
from PIL import Image
from openai import OpenAI
from typing import Dict, Any, Tuple
from argparse import ArgumentParser
from torch.utils.data import Dataset
from tqdm.contrib.concurrent import thread_map
from transformers import CLIPProcessor, SiglipVisionModel
from ssr.utils.prompt import SSRSpecialToken, string_truncation
from ssr.utils.misc import get_chunk, change_ext, convert_depth
from ssr.models.tokenization_internlm3 import Internlm3Tokenizer


class SpatialQACoTDataset(Dataset):
    def __init__(
        self
        , data_dir: str
        , tokenizer: Internlm3Tokenizer
        , max_length: Tuple[int, int, int]
        , clip_processor: CLIPProcessor
        , siglip_processor: SiglipVisionModel
    ) -> None:
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.clip_processor = clip_processor
        self.siglip_processor = siglip_processor
        self.data = json.load(open(os.path.join(data_dir, "ssr_spatialqa.json"), "r"))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            item = self.data[idx]
            question, rationale, answer = item["question"], item["rationale"], item["answer"]
            question, rationale, answer = (string_truncation(text, self.tokenizer, max_len) for text, max_len in zip((question, rationale, answer), self.max_length))
            question = "\n".join([SSRSpecialToken.IMAGE_TOKEN, SSRSpecialToken.DEPTH_TOKEN, question])
            image_path = os.path.join(self.data_dir, "images", item["image_path"])
            image = Image.open(image_path).convert("RGB")
            image = (self.clip_processor(images=image, return_tensors="pt").pixel_values).squeeze(0)
            depth_path = os.sep.join([f"{item['image_path'].split(os.sep)[0]}_d"] + item['image_path'].split(os.sep)[1:])
            depth_path = os.path.join(self.data_dir, "images", depth_path)
            depth_path = change_ext(depth_path, "png")
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


GEN_CoT_PROMPT = lambda question, answer: f"""I have an image and a question that I want you to answer.
I need you to strictly follow the format with four specific sections: summary, caption, reasoning, and conclusion.
It is crucial that you adhere to this structure exactly as outlined and that the final answer in the conclusion matches the standard correct answer precisely.
To explain further:
    - In summary, briefly explain what steps you'll take to solve the problem.
    - In caption, describe the contents of the image, specifically focusing on details relevant to the question.
    - In reasoning, outline a step-by-step thought process you would use to solve the problem based on the image.
    - In conclusion, give the final answer in a direct format, and it must match the correct answer exactly. If it's a multiple choice question, the conclusion should only include the option without repeating what the option is.
Finally, integrate these sections into a natural thinking paragraph.

Here's the question and answer:
Question: {question}
Answer: {answer}"""


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "SpatialQA"))
    parser.add_argument("--mode", type=str, default="preprocess", choices=["preprocess", "gen_rationale"])
    parser.add_argument("--num_chunk", type=int, default=4)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--max_workers", type=int, default=30)
    args = parser.parse_args()

    if args.mode == "preprocess" and not os.path.exists(os.path.join(args.data_dir, "SSR_SpatialQA.json")):
        with open(os.path.join(args.data_dir, "SpatialQA.json"), "r") as file:
            raw_data = json.load(file)
        data = []
        for raw_item in tqdm(raw_data, desc="Convert SpatialQA", ncols=100):
            if "image" in raw_item and len(raw_item["image"]) == 1 and "conversations" in raw_item and len(raw_item["conversations"]) % 2 == 0:
                item = raw_item.copy()
                item["image"] = item["image"][0]
                question = item["conversations"][0]["value"]
                answer = item["conversations"][1]["value"]
                question = question.replace("<image 1>\n", "")
                item["question"] = question
                item["answer"] = answer
                data.append(item)
        with open(os.path.join(args.data_dir, "SSR_SpatialQA.json"), "w") as file:
            json.dump(data, file, indent=4)
    elif args.mode == "gen_rationale":
        gpt = OpenAI(base_url="https://vip.apiyi.com/v1", api_key="sk-2SP6kIIJ2Xg9lLN651Ad7aBb6d644a46B59f525b3bB5C6A6")

        def encode_image(image_path: str) -> str:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")

        spatialqa_cot_data = []
        def gen_rationale(item: Dict[str, Any]) -> str:
            image_path, question, answer = (item[key] for key in ["image", "question", "answer"])
            image_path = os.path.join(args.data_dir, "images", image_path)
            try:
                completion = gpt.chat.completions.create(
                    model="gpt-4o-mini"
                    , messages=[{
                        "role": "user"
                        , "content": [
                            {"type": "text", "text": f"{GEN_CoT_PROMPT(question, answer)}"}
                            , {
                                "type":"image_url"
                                , "image_url": {
                                    "url": f"data:image/png;base64,{encode_image(image_path)}"
                                }
                            }
                        ]
                    }]
                )
                rationale = completion.choices[0].message.content
            except Exception as e:
                print(e)
                return
            spatialqa_cot_data.append({
                "question": question
                , "rationale": rationale
                , "answer": answer
                , "image_path": item["image"]
            })
        
        with open(os.path.join(args.data_dir, "SSR_SpatialQA.json"), "r") as file:
            spatial_qa_data = json.load(file)
        spatial_qa_data = get_chunk(spatial_qa_data, args.num_chunk, args.chunk_idx)
        
        thread_map(
            gen_rationale
            , spatial_qa_data
            , max_workers=args.max_workers
            , desc="Generate SpatialQA Rationale"
        )
        with open(os.path.join(args.data_dir, f"SpatialQA_CoT_{args.chunk_idx}_{args.num_chunk}.json"), "w") as file:
            json.dump(spatialqa_cot_data, file, indent=4)