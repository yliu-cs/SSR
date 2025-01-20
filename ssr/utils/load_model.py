import torch
from typing import Tuple
from transformers import BitsAndBytesConfig
from ssr.models.modeling_ssr import SSRForCausalLM
from ssr.models.modeling_smamba import SSRMambaForCausalLM
from ssr.models.tokenization_internlm3 import InternLM3Tokenizer


def load_internlm3(link: str, bits: int) -> Tuple[SSRForCausalLM, InternLM3Tokenizer]:
    # huggingface model configuration
    huggingface_config = {}
    # Bit quantization
    if bits in [4, 8]:
        huggingface_config.update(dict(
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=bits == 4,
                load_in_8bit=bits == 8,
                llm_int8_skip_modules=["vit", "vision_proj", "output", "ffn"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
        ))
    else:
        huggingface_config.update(dict(
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        ))
    # loading backbone model
    ssr = SSRForCausalLM.from_pretrained(link, **huggingface_config)
    # loading ssr tokenizer
    # adding <image> and <tor> special token
    ssr_tokenizer = InternLM3Tokenizer.from_pretrained(link, padding_side='left')
    ssr_tokenizer.add_tokens("<image>", special_tokens=True)
    ssr_tokenizer.add_tokens("<depth>", special_tokens=True)
    ssr_tokenizer.add_tokens("<tor>", special_tokens=True)
    return ssr, ssr_tokenizer


def load_smamba(link: str) -> SSRMambaForCausalLM:
    # huggingface model configuration
    huggingface_config = {}
    huggingface_config.update(dict(
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ))
    # SSR Mamba Model (no fp32)
    mmamba = SSRMambaForCausalLM.from_pretrained(link, **huggingface_config)
    return mmamba