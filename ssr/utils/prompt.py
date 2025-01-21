import re
from typing import List


TOR_TOKEN = "<tor>"


def find_special_token_indices(main_string: str, special_token: str) -> List[int]:
    return [match.start() for match in re.finditer(re.escape(special_token), main_string)]


def custom_join(lst: List[str]) -> str:
    if not lst:
        return ""
    it = iter(lst)
    result = [next(it)]
    prev_is_tor = (result[0] == TOR_TOKEN)
    for elem in it:
        if prev_is_tor or elem == TOR_TOKEN:
            result.append(elem)
        else:
            result.append(' ' + elem)
        prev_is_tor = (elem == TOR_TOKEN)
    return ''.join(result)


def insert_tor(sentence: str, n_tor: int) -> str:
    parts = sentence.split()
    if n_tor == 0:
        return sentence
    positions = []
    if len(parts) + 1 == 0:
        return " ".join([TOR_TOKEN] * n_tor)
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
        new_elements.extend([TOR_TOKEN] * count.get(i, 0))
        if i < len(parts):
            new_elements.append(parts[i])
    if not new_elements and sentence.strip() == "" and n_tor > 0:
        new_elements = [TOR_TOKEN] * n_tor
    return custom_join(new_elements)


def repeat_special_token(input_string: str, special_token: str, n_repeat: int) -> str:
    if special_token not in input_string:
        return input_string
    result = input_string.replace(special_token, special_token * n_repeat)
    return result


def construct_mamba_conversation(
    question: str
    , rationale: str = ""
    , n_tor: int = 10
) -> str:
    messages = [
        {"role": "user", "content": question}
        , {"role": "rationale", "content": insert_tor(rationale, n_tor)}
    ]
    conv = "\n".join([f"[UNUSED_TOKEN_146]{msg['role']}:\n{msg['content']}[UNUSED_TOKEN_145]" for msg in messages])
    conv = "<s>" + conv + "</s>"
    return conv


def construct_internlm_conversation(
    question: str
    , answer: str = ""
    , n_tor: int = 10
) -> str:
    system_prompt = "You should give helpful answer to user based on the rationale."
    messages = [
        {"role": "system", "content": system_prompt}
        , {"role": "user", "content": question}
        , {"role": "assistant", "content": answer if answer else ""}
    ]
    conv = "\n".join([f"[UNUSED_TOKEN_146]{msg['role']}:\n{msg['content']}[UNUSED_TOKEN_145]" for msg in messages])
    conv = "<s>" + insert_tor("", n_tor) + conv + "</s>"
    return conv


if __name__ == "__main__":
    question = "Formulate an answer to this elaborate question: When was the band that released the single Ignorance formed ?"
    rationale = "The band that released the single Ignorance is Paramore and they were formed in 2004."
    print(f"{insert_tor(rationale, n_tor=10)}")
    answer = "2004"
    print(f"{construct_mamba_conversation(question=question, rationale=rationale)=}")
    print(f"{construct_mamba_conversation(question=question, n_tor=10)=}")
    print(f"{construct_internlm_conversation(question=question, answer=answer, n_tor=10)=}")
    print(f"{construct_internlm_conversation(question=question, n_tor=10)=}")