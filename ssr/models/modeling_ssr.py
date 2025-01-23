import torch
from torch import nn
from torch.amp import autocast
from ssr.utils.misc import freeze_module, has_nan
from ssr.utils.prompt import SSRStage, SSRSpecialToken
from ssr.utils.load_ptm import load_mamba, load_internlm3
from transformers import PretrainedConfig, PreTrainedModel, CLIPVisionModel, SiglipVisionModel


class SSRConfig(PretrainedConfig):
    model_type = "SSR"
    def __init__(
        self
        , mamba_path: str
        , internlm3_path: str
        , bits: int
        , stage: SSRStage
        , device: torch.device
        , **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.mamba_path = mamba_path
        self.internlm3_path = internlm3_path
        self.bits = bits
        self.stage = stage
        self.device = device


class SSR(PreTrainedModel):
    def __init__(self, config: SSRConfig, clip_vision: CLIPVisionModel, siglip: SiglipVisionModel) -> None:
        super().__init__(config)
        self.config = config
        self.image_encoder = clip_vision
        self.depth_encoder = siglip
        self.mamba = load_mamba(self.config.mamba_path, device=self.config.device)
        self.internlm3, self.tokenizer = load_internlm3(self.config.internlm3_path, bits=self.config.bits, device=self.config.device)
        self.mamba.backbone.embeddings = nn.Embedding(num_embeddings=len(self.tokenizer) + 2, embedding_dim=self.mamba.config.hidden_size)
        self.image_token_id, self.depth_token_id, self.tor_token_id = self.tokenizer.convert_tokens_to_ids(
            [SSRSpecialToken.IMAGE_TOKEN, SSRSpecialToken.DEPTH_TOKEN, SSRSpecialToken.TOR_TOKEN]
        )
        self.mamba.image_token_id, self.mamba.depth_token_id, self.mamba.tor_token_id = self.image_token_id, self.depth_token_id, self.tor_token_id
        self.internlm3.image_token_id, self.internlm3.depth_token_id, self.internlm3.tor_token_id = self.image_token_id, self.depth_token_id, self.tor_token_id
        freeze_module(self.image_encoder)
        freeze_module(self.depth_encoder)
        if self.config.stage == SSRStage.mamba:
            freeze_module(self.internlm3)
    
    def forward(
        self
        , input_ids: torch.Tensor
        , attention_mask: torch.Tensor
        , labels: torch.Tensor
        , image: torch.Tensor
        , depth: torch.Tensor
    ):
        print(f"{input_ids.size()=} {input_ids.dtype=} {has_nan(input_ids)=}")
        print(f"{attention_mask.size()=} {attention_mask.dtype=} {has_nan(attention_mask)=}")
        print(f"{labels.size()=} {labels.dtype=} {has_nan(labels)=}")
        print(f"{image.size()=} {image.dtype=} {has_nan(image)=}")
        print(f"{depth.size()=} {depth.dtype=} {has_nan(depth)=}")
        print("-" * 50 + "Input Out" + "-" * 50)
        image_mask = torch.zeros_like(input_ids).bool()
        image_mask[torch.where(input_ids == self.image_token_id)] = True
        image_embeds = self.image_encoder(image).last_hidden_state[:, 1:, :]
        print(f"{image_embeds.size()=} {image_embeds.dtype=} {has_nan(image_embeds)=}")
        depth_embeds = self.depth_encoder(depth).last_hidden_state
        print(f"{depth_embeds.size()=} {depth_embeds.dtype=} {has_nan(depth_embeds)=}")
        with autocast("cuda", enabled=False):
            mamba_outputs = self.mamba(
                input_ids=input_ids
                , attention_mask=attention_mask
                , image_embeds=image_embeds
                , depth_embeds=depth_embeds
            )
        tor_embeds = mamba_outputs.tor_embeds
        print(f"{tor_embeds.size()=} {tor_embeds.dtype=} {has_nan(tor_embeds)=}")
        print("-" * 50 + "Mamba Out" + "-" * 50)
        with autocast("cuda", enabled=True, dtype=torch.float16):
            internlm_outputs = self.internlm3(
                input_ids=input_ids
                , image_embeds=image_embeds
                , depth_embeds=depth_embeds
                , tor_embeds=tor_embeds
                , attention_mask=attention_mask
                , image_mask=image_mask
                , labels=labels
            )
        print(f"{internlm_outputs.loss=} {internlm_outputs.tor_embeds.dtype=}")
        exit()