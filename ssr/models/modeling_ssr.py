import os
import torch
from torch import nn
from einops import rearrange
from torch.amp import autocast
from typing import Union, Tuple, Dict
from ssr.utils.misc import freeze_module
from ssr.utils.misc import build_projector
from ssr.utils.prompt import SSRSpecialToken
from ssr.utils.load_ptm import load_mamba, load_internlm3
from .modeling_sinternlm3 import SSRCausalLMOutputWithPast
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import PretrainedConfig, PreTrainedModel, CLIPVisionModel, SiglipVisionModel


class ProjectorConfig(PretrainedConfig):
    model_type = "Projector"
    def __init__(
        self
        , mm_hidden_size: int = 1024
        , hidden_size: int = 4096
        , **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.mm_hidden_size = mm_hidden_size
        self.hidden_size = hidden_size


class Projector(PreTrainedModel):
    config_class = ProjectorConfig
    def __init__(self, config: ProjectorConfig) -> None:
        super().__init__(config)
        self.projector = build_projector(config.mm_hidden_size, config.hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class SSRConfig(PretrainedConfig):
    model_type = "SSR"
    def __init__(
        self
        , mamba_path: str = ""
        , internlm3_path: str = ""
        , stage1_path: str = ""
        , stage: int = 1
        , bits: int = 4
        , lora_r: int = 64
        , lora_alpha: int = 64
        , lora_dropout: float = 0.1
        , **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.mamba_path = mamba_path
        self.internlm3_path = internlm3_path
        self.stage1_path = stage1_path
        self.stage = stage
        self.bits = bits
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout


class SSR(PreTrainedModel):
    config_class = SSRConfig
    def __init__(self, config: SSRConfig, clip_vision: CLIPVisionModel, siglip: SiglipVisionModel) -> None:
        super().__init__(config)
        self.config = config
        self.image_encoder = clip_vision
        self.depth_encoder = siglip
        if self.config.stage == 1:
            self.mamba = load_mamba(self.config.mamba_path)
            self.internlm3, self.tokenizer = load_internlm3(self.config.internlm3_path, bits=self.config.bits, add_special_tokens=True)
            self.mamba.backbone.embeddings = nn.Embedding(num_embeddings=self.internlm3.config.vocab_size, embedding_dim=self.mamba.config.hidden_size)
            self.mamba.config.vocab_size = self.internlm3.config.vocab_size
            self.mamba_image_proj = Projector(ProjectorConfig(self.image_encoder.config.hidden_size, self.mamba.config.hidden_size))
            self.mamba_depth_proj = Projector(ProjectorConfig(self.depth_encoder.config.hidden_size, self.mamba.config.hidden_size))
            self.tor_proj = Projector(ProjectorConfig(self.mamba.config.hidden_size, self.internlm3.config.hidden_size))
            self.internlm3_image_proj = Projector(ProjectorConfig(self.image_encoder.config.hidden_size, self.internlm3.config.hidden_size))
            self.internlm3_depth_proj = Projector(ProjectorConfig(self.depth_encoder.config.hidden_size, self.internlm3.config.hidden_size))
        else:
            pretrained_path = os.path.join(getattr(self.config, f"stage{self.config.stage - 1}_path"))
            self.mamba = load_mamba(os.path.join(pretrained_path, "mamba"))
            self.internlm3, self.tokenizer = load_internlm3(os.path.join(pretrained_path, "internlm3"), bits=self.config.bits, add_special_tokens=False)
            self.mamba_image_proj = Projector.from_pretrained(os.path.join(pretrained_path, "mamba_image_proj"))
            self.mamba_depth_proj = Projector.from_pretrained(os.path.join(pretrained_path, "mamba_depth_proj"))
            self.tor_proj = Projector.from_pretrained(os.path.join(pretrained_path, "tor_proj"))
            self.internlm3_image_proj = Projector.from_pretrained(os.path.join(pretrained_path, "internlm3_image_proj"))
            self.internlm3_depth_proj = Projector.from_pretrained(os.path.join(pretrained_path, "internlm3_depth_proj"))
        self.image_token_id, self.depth_token_id, self.tor_token_id = self.tokenizer.convert_tokens_to_ids(
            [SSRSpecialToken.IMAGE_TOKEN, SSRSpecialToken.DEPTH_TOKEN, SSRSpecialToken.TOR_TOKEN]
        )
        self.mamba.image_token_id, self.mamba.depth_token_id = self.image_token_id, self.depth_token_id
        self.internlm3.image_token_id, self.internlm3.depth_token_id, self.internlm3.tor_token_id = self.image_token_id, self.depth_token_id, self.tor_token_id

    def prepare_modules(self) -> None:
        freeze_module(self.image_encoder)
        freeze_module(self.depth_encoder)
        if self.config.stage == 1:
            freeze_module(self.internlm3)
        elif self.config.stage == 2:
            self.internlm3 = prepare_model_for_kbit_training(self.internlm3)
            lora_config = LoraConfig(
                r=self.config.lora_r
                , lora_alpha=self.config.lora_alpha
                , target_modules="all-linear"
                , lora_dropout=self.config.lora_dropout
                , bias="none"
                , task_type="CAUSAL_LM"
            )
            self.internlm3 = get_peft_model(self.internlm3, lora_config)
    
    def forward(
        self
        , input_ids: torch.Tensor
        , attention_mask: torch.Tensor
        , labels: torch.Tensor
        , image: torch.Tensor
        , depth: torch.Tensor
    ) -> Union[Tuple, SSRCausalLMOutputWithPast]:
        image_embeds = self.image_encoder(image).hidden_states[-2][:, 1:, :]
        depth_embeds = self.depth_encoder(depth).hidden_states[-1][:, :, :]
        # print(f"{input_ids.size()=} {attention_mask.size()=} {labels.size()=} {image_embeds.size()=} {depth_embeds.size()=}")
        mamba_outputs = self.mamba(
            input_ids=input_ids.clone()
            , attention_mask=attention_mask
            , image_embeds=self.mamba_image_proj(image_embeds.clone())
            , depth_embeds=self.mamba_depth_proj(depth_embeds.clone())
        )
        last_hidden_state = mamba_outputs.last_hidden_state
        tor_embeds = self.tor_proj(last_hidden_state[(input_ids == self.tor_token_id), :])
        tor_embeds = rearrange(tor_embeds, f"(b l) d -> b l d", b=last_hidden_state.size(0))
        internlm_outputs = self.internlm3(
            input_ids=input_ids.clone()
            , image_embeds=self.internlm3_image_proj(image_embeds.clone())
            , depth_embeds=self.internlm3_depth_proj(depth_embeds.clone())
            , tor_embeds=tor_embeds
            , attention_mask=attention_mask
            , labels=labels
        )
        return internlm_outputs