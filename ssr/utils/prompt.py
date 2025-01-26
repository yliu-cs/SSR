import re
import torch
from typing import List
from enum import Enum, StrEnum
from ssr.models.tokenization_internlm3 import Internlm3Tokenizer


class SSRStage(Enum):
    mamba = 1
    internlm = 2


class SSRSpecialToken(StrEnum):
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    BOR_TOKEN = "<|im_start|>"
    EOR_TOKEN = "<|im_end|>"
    TOR_TOKEN = "<tor>"
    IMAGE_TOKEN = "<image>"
    DEPTH_TOKEN = "<depth>"

IGNORE_TOKEN_ID = -100
SYSTEM_PROMPT = "\n".join([
    "You are an AI assistant whose name is InternLM."
    , "- InternLM is a conversational language model that is designed to be helpful, honest, and harmless."
    , "- InternLM can understand and communicate fluently."
])


def find_special_token_indices(main_string: str, special_token: str) -> List[int]:
    return [match.start() for match in re.finditer(re.escape(special_token), main_string)]


def custom_join(lst: List[str]) -> str:
    if not lst:
        return ""
    it = iter(lst)
    result = [next(it)]
    prev_is_tor = (result[0] == SSRSpecialToken.TOR_TOKEN)
    for elem in it:
        if prev_is_tor or elem == SSRSpecialToken.TOR_TOKEN:
            result.append(elem)
        else:
            result.append(" " + elem)
        prev_is_tor = (elem == SSRSpecialToken.TOR_TOKEN)
    return "".join(result)


def insert_tor(sentence: str, n_tor: int) -> str:
    parts = sentence.split()
    if n_tor == 0:
        return sentence
    positions = []
    if len(parts) + 1 == 0:
        return " ".join([SSRSpecialToken.TOR_TOKEN] * n_tor)
    elif n_tor == 1:
        pos = round((len(parts) + 1 - 1) / 2)
        positions.append(pos)
    else:
        for i in range(n_tor):
            pos = round(i * (len(parts) + 1 - 1) / (n_tor - 1))
            positions.append(pos)
    count = {}
    for pos in positions:
        count[pos] = count.get(pos, 0) + 1
    new_elements = []
    for i in range(len(parts) + 1):
        new_elements.extend([SSRSpecialToken.TOR_TOKEN] * count.get(i, 0))
        if i < len(parts):
            new_elements.append(parts[i])
    if not new_elements and sentence.strip() == "" and n_tor > 0:
        new_elements = [SSRSpecialToken.TOR_TOKEN] * n_tor
    return custom_join(new_elements)


def repeat_special_token(input_string: str, special_token: str, n_repeat: int) -> str:
    if special_token not in input_string:
        return input_string
    result = input_string.replace(special_token, special_token * n_repeat)
    return result


def repeat_special_tokens(input_string: str, special_tokens: List[str], n_repeats: List[int]) -> str:
    result = input_string
    for special_token, n_repeat in zip(special_tokens, n_repeats):
        result = repeat_special_token(result, special_token, n_repeat)
    return result


def construct_conversation(
    question: str
    , rationale: str = ""
    , answer: str = ""
    , stage: SSRStage = SSRStage.mamba
    , n_tor: int = 10
) -> str:
    messages = []
    if stage != SSRStage.mamba:
        messages.append({"role": "system", "content": "\n".join([SYSTEM_PROMPT, "You should give helpful answer to user based on the rationale."])})
    messages.append({"role": "user", "content": question})
    messages.append({"role": "rationale", "content": insert_tor(rationale, n_tor)})
    if stage == SSRStage.internlm:
        messages.append({"role": "assistant", "content": answer})
    conv = "\n".join([f"{SSRSpecialToken.BOR_TOKEN}{msg['role']}\n{msg['content']}{SSRSpecialToken.EOR_TOKEN}" for msg in messages])
    conv = SSRSpecialToken.BOS_TOKEN + conv + SSRSpecialToken.EOS_TOKEN
    return conv


def create_labels(
    input_ids: torch.Tensor
    , stage: SSRStage
    , tokenizer: Internlm3Tokenizer
) -> str:
    target = torch.tensor(
        tokenizer.convert_tokens_to_ids(["<|im_start|>", "rational", "e", "\n"] if stage == SSRStage.mamba else ["<|im_start|>", "assistant", "\n"])
        , device=input_ids.device
        , dtype=input_ids.dtype
    )
    tor_id = tokenizer.convert_tokens_to_ids([SSRSpecialToken.TOR_TOKEN])[0]
    labels = torch.full_like(input_ids, IGNORE_TOKEN_ID)
    for batch_idx in range(input_ids.size(0)):
        for i in range(input_ids.size(1) - target.size(0)):
            if torch.equal(input_ids[batch_idx, i:i + target.size(0)], target):
                labels[batch_idx, i:] = input_ids[batch_idx, i:].clone()
                break
    labels[labels == tor_id] = IGNORE_TOKEN_ID
    return labels


if __name__ == "__main__":
    question = "Formulate an answer to this elaborate question: When was the band that released the single Ignorance formed ?"
    rationale = "The band that released the single Ignorance is Paramore and they were formed in 2004."
    answer = "2004"
    # print(f"{insert_tor(rationale, n_tor=10)}")
    # print(f"{insert_tor('', n_tor=10)}")
    print(f"{construct_conversation(question=question, rationale=rationale, answer=answer, stage=SSRStage.mamba, n_tor=10)=}")
    print(f"{construct_conversation(question=question, rationale=rationale, answer=answer, stage=SSRStage.internlm, n_tor=10)=}")