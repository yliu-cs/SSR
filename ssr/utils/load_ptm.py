import os
import torch
import depth_pro
from typing import Tuple
from depth_pro.depth_pro import DepthPro
from ssr.utils.misc import freeze_module
from torchvision.transforms import Compose
from ssr.utils.prompt import SSRSpecialToken
from ssr.models.modeling_smamba import SSRMambaForCausalLM
from ssr.models.tokenization_internlm3 import Internlm3Tokenizer
from ssr.models.modeling_sinternlm3 import SSRInternlm3ForCausalLM
from transformers import BitsAndBytesConfig, CLIPProcessor, CLIPVisionModel, SiglipProcessor, SiglipVisionModel


def load_internlm3(model_path: str, bits: int, device: torch.device) -> Tuple[SSRInternlm3ForCausalLM, Internlm3Tokenizer]:
    # Huggingface Model Configuration with Bit quantization
    if bits in [4, 8]:
        huggingface_config = {
            "torch_dtype": torch.bfloat16
            , "low_cpu_mem_usage": True
            , "attn_implementation": "flash_attention_2"
            , "quantization_config": BitsAndBytesConfig(
                load_in_4bit=bits == 4
                , load_in_8bit=bits == 8
                , llm_int8_skip_modules=["output", "ffn"]
                , llm_int8_threshold=6.0
                , llm_int8_has_fp16_weight=False
                , bnb_4bit_compute_dtype=torch.bfloat16
                , bnb_4bit_use_double_quant=True
                , bnb_4bit_quant_type="nf4"
                , bnb_4bit_quant_storage=torch.bfloat16
            )
        }
    else:
        huggingface_config = {
            "torch_dtype": torch.bfloat16
            , "low_cpu_mem_usage": True
            , "attn_implementation": "flash_attention_2"
        }
    # Loading Backbone Model
    ssr = SSRInternlm3ForCausalLM.from_pretrained(model_path, **huggingface_config, device_map=device)
    # Loading SSR Tokenizer & Adding <image> & <depth> & <tor> Special Token
    ssr_tokenizer = Internlm3Tokenizer.from_pretrained(model_path, padding_side="left")
    ssr_tokenizer.add_tokens(SSRSpecialToken.IMAGE_TOKEN, special_tokens=True)
    ssr_tokenizer.add_tokens(SSRSpecialToken.DEPTH_TOKEN, special_tokens=True)
    ssr_tokenizer.add_tokens(SSRSpecialToken.TOR_TOKEN, special_tokens=True)
    return ssr, ssr_tokenizer


def load_mamba(model_path: str, device: torch.device) -> SSRMambaForCausalLM:
    # Huggingface Model Configuration
    huggingface_config = {
        "ignore_mismatched_sizes": True
        , "torch_dtype": torch.float32
        , "low_cpu_mem_usage": True
    }
    # SSR Mamba Model (no fp32)
    smamba = SSRMambaForCausalLM.from_pretrained(model_path, **huggingface_config, device_map=device)
    return smamba


def load_clip_vit(model_path: str, device: torch.device) -> Tuple[CLIPProcessor, CLIPVisionModel]:
    return (
        CLIPProcessor.from_pretrained(model_path)
        , CLIPVisionModel.from_pretrained(model_path, output_hidden_states=True, device_map=device)
    )


def load_siglip(model_path: str, device: torch.device) -> Tuple[SiglipProcessor, SiglipVisionModel]:
    return (
        SiglipProcessor.from_pretrained(model_path)
        , SiglipVisionModel.from_pretrained(model_path, output_hidden_states=True, device_map=device)
    )


def load_depth_pro(model_path: str, device: torch.device) -> Tuple[DepthPro, Compose]:
    model, transform = depth_pro.create_model_and_transforms(
        device=device
        , checkpoint_uri=os.path.join(model_path, "depth_pro.pt")
    )
    freeze_module(model)
    model.eval()
    return model, transform