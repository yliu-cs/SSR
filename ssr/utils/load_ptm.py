import os
import torch
import depth_pro
from typing import Tuple
from ssr.models.modeling_smamba import SSRMambaForCausalLM
from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
from ssr.models.tokenization_internlm3 import InternLM3Tokenizer
from ssr.models.modeling_sinternlm3 import SSRInternLM3ForCausalLM
from transformers import BitsAndBytesConfig, CLIPProcessor, CLIPVisionModel, SiglipProcessor, SiglipVisionModel


def load_internlm3(model_path: str, bits: int) -> Tuple[SSRInternLM3ForCausalLM, InternLM3Tokenizer]:
    # Huggingface Model Configuration
    huggingface_config = {}
    # Bit quantization
    if bits in [4, 8]:
        huggingface_config.update(dict(
            torch_dtype=torch.float16
            , low_cpu_mem_usage=True
            , attn_implementation="flash_attention_2"
            , quantization_config=BitsAndBytesConfig(
                load_in_4bit=bits == 4
                , load_in_8bit=bits == 8
                , llm_int8_skip_modules=["image_proj", "depth_proj", "output", "ffn"]
                , llm_int8_threshold=6.0
                , llm_int8_has_fp16_weight=False
                , bnb_4bit_compute_dtype=torch.float16
                , bnb_4bit_use_double_quant=True
                , bnb_4bit_quant_type="nf4"
            )
        ))
    else:
        huggingface_config.update(dict(
            torch_dtype=torch.float16
            , low_cpu_mem_usage=True
            , attn_implementation="flash_attention_2"
        ))
    # Loading Backbone Model
    ssr = SSRInternLM3ForCausalLM.from_pretrained(model_path, **huggingface_config)
    # Loading SSR Tokenizer & Adding <image> And <tor> Special Token
    ssr_tokenizer = InternLM3Tokenizer.from_pretrained(model_path, padding_side="left")
    ssr_tokenizer.add_tokens("<image>", special_tokens=True)
    ssr_tokenizer.add_tokens("<depth>", special_tokens=True)
    ssr_tokenizer.add_tokens("<tor>", special_tokens=True)
    return ssr, ssr_tokenizer


def load_mamba(model_path: str) -> SSRMambaForCausalLM:
    # Huggingface Model Configuration
    huggingface_config = {}
    huggingface_config.update(dict(
        ignore_mismatched_sizes=True
        , torch_dtype=torch.float32
        , low_cpu_mem_usage=True
    ))
    # SSR Mamba Model (no fp32)
    smamba = SSRMambaForCausalLM.from_pretrained(model_path, **huggingface_config)
    return smamba


def load_clip_vit(model_path: str) -> Tuple[CLIPProcessor, CLIPVisionModel]:
    return CLIPProcessor.from_pretrained(model_path), CLIPVisionModel.from_pretrained(model_path)


def load_siglip(model_path: str) -> Tuple[SiglipProcessor, SiglipVisionModel]:
    return SiglipProcessor.from_pretrained(model_path), SiglipVisionModel.from_pretrained(model_path)


def load_depth_pro(model_path: str):
    model, transform = depth_pro.create_model_and_transforms(checkpoint_uri=os.path.join(model_path, "depth_pro.pt"))
    model.eval()
    return model, transform