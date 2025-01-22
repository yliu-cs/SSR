import torch
from torch import nn
from ssr.utils.load_ptm import load_mamba, load_internlm3
from transformers import PretrainedConfig, PreTrainedModel, CLIPVisionModel, SiglipVisionModel


class SSRConfig(PretrainedConfig):
    model_type = "SSR"
    def __init__(
        self
        , mamba_path: str
        , internlm3_path: str
        , bits: str
        , device: torch.device
        , **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.mamba_path = mamba_path
        self.internlm3_path = internlm3_path
        self.bits = bits
        self.device = device


class SSR(PreTrainedModel):
    def __init__(self, config: SSRConfig, clip_vision: CLIPVisionModel, siglip: SiglipVisionModel) -> None:
        super().__init__(config)
        self.config = config
        self.image_encoder = clip_vision
        self.depth_encoder = siglip
        self.mamba = load_mamba(self.config.mamba_path, device=self.config.device)
        self.internlm3, self.tokenizer = load_internlm3(self.config.internlm3_path, self.config.bits, device=self.config.device)
        self.mamba.backbone.embeddings = nn.Embedding(num_embeddings=len(self.tokenizer) + 2, embedding_dim=self.mamba.config.hidden_size)
    
    def forward(
        self
        , input_ids: torch.Tensor
        , attention_mask: torch.Tensor
        , labels: torch.Tensor
        , image: torch.Tensor
        , depth: torch.Tensor
    ):
        print(f"{input_ids.size()=} {attention_mask.size()=} {labels.size()=} {image.size()=} {depth.size()=}")
        exit()