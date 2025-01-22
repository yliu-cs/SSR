import re
import torch
from dataclasses import dataclass
from typing import Tuple, Optional, Union
from transformers import MambaForCausalLM
from ssr.utils.misc import build_projector
from transformers.modeling_outputs import ModelOutput


class MambaCache:
    def __init__(self, config, batch_size, dtype=torch.float16, device=None) -> None:
        self.seqlen_offset = 0
        self.dtype = dtype
        intermediate_size = config.intermediate_size
        ssm_state_size = config.state_size
        conv_kernel_size = config.conv_kernel
        self.conv_states = {
            i: torch.zeros(batch_size, intermediate_size, conv_kernel_size, device=device, dtype=dtype)
            for i in range(config.num_hidden_layers)
        }
        self.ssm_states = {
            i: torch.zeros(batch_size, intermediate_size, ssm_state_size, device=device, dtype=dtype)
            for i in range(config.num_hidden_layers)
        }


@dataclass
class MambaCausalLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    cache_params: Optional[MambaCache] = None
    tor_embeds: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class SSRMambaForCausalLM(MambaForCausalLM):
    def __init__(self, config) -> None:
        super().__init__(config)
        # Initialize projectors for Image and Tor
        self.image_proj = build_projector(1024, self.config.hidden_size)
        self.tor_proj = build_projector(self.config.hidden_size, 4096)

    def merge_input_embeds_with_image(
        self
        , image_embeds: torch.FloatTensor
        , depth_embeds: torch.FloatTensor
        , inputs_embeds: torch.FloatTensor
        , input_ids: torch.LongTensor
    ) -> None:
        # Merge Image Embeds
        if image_embeds is not None and input_ids.size(1) != 1:
            image_embeds = self.image_proj(image_embeds.to(inputs_embeds.dtype))
            batch_idx_image_embeds = 0
            B, C, D = image_embeds.size()
            for batch_idx, input_id in enumerate(input_ids):
                matching = torch.where(input_id == self.config.image_token_id)
                num_image_tokens_per_sample = len(matching[0]) // C
                inputs_embeds[batch_idx][matching] = image_embeds[batch_idx_image_embeds: batch_idx_image_embeds + num_image_tokens_per_sample].view(-1, D)
                batch_idx_image_embeds += num_image_tokens_per_sample
        # Merge Depth Embeds
        if depth_embeds is not None and input_ids.shape[1] != 1:
            depth_embeds = self.depth_proj(depth_embeds.to(inputs_embeds.dtype))
            batch_idx_depth_embeds = 0
            B, C, D = depth_embeds.size()
            for batch_idx, input_id in enumerate(input_ids):
                matching = torch.where(input_id == self.config.depth_token_id)
                num_depth_tokens_per_sample = len(matching[0]) // C
                inputs_embeds[batch_idx][matching] = depth_embeds[batch_idx_depth_embeds: batch_idx_depth_embeds + num_depth_tokens_per_sample].view(-1, D)
                batch_idx_depth_embeds += num_depth_tokens_per_sample
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_embeds: torch.FloatTensor = None,
        depth_embeds: torch.FloatTensor = None,
        cache_params: Optional[MambaCache] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, MambaCausalLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            self.merge_input_embeds_with_image_depth(image_embeds, depth_embeds, inputs_embeds, input_ids)

        mamba_outputs = self.backbone(
            input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
        )
        hidden_states = mamba_outputs[0]
        
        return MambaCausalLMOutput(
            loss=None,
            cache_params=mamba_outputs.cache_params,
            tor_embeds=self.tor_proj(hidden_states[torch.where(input_ids == self.config.tor_token_index)]),
            hidden_states=mamba_outputs.hidden_states,
        )