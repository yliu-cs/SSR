import torch
from dataclasses import dataclass
from typing import Tuple, Optional, Union
from transformers import MambaForCausalLM
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
    last_hidden_state: Optional[Tuple[torch.FloatTensor]] = None


class SSRMambaForCausalLM(MambaForCausalLM):
    def __init__(self, config) -> None:
        super().__init__(config)

    def merge_input_embeds_with_image_depth(
        self
        , image_embeds: torch.FloatTensor
        , depth_embeds: torch.FloatTensor
        , inputs_embeds: torch.FloatTensor
        , input_ids: torch.LongTensor
    ) -> None:
        # Merge Image Embeds
        if image_embeds is not None and input_ids.size(1) != 1:
            for batch_idx, input_id in enumerate(input_ids):
                matching = torch.where(input_id == self.image_token_id)
                inputs_embeds[batch_idx][matching] = image_embeds[batch_idx, ...]
        # Merge Depth Embeds
        if depth_embeds is not None and input_ids.shape[1] != 1:
            for batch_idx, input_id in enumerate(input_ids):
                matching = torch.where(input_id == self.depth_token_id)
                inputs_embeds[batch_idx][matching] = depth_embeds[batch_idx, ...]
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
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
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        last_hidden_state = mamba_outputs.last_hidden_state

        return MambaCausalLMOutput(
            loss=None,
            cache_params=mamba_outputs.cache_params,
            last_hidden_state=last_hidden_state,
        )