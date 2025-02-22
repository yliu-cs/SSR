import torch
from typing import Tuple, Optional, Union
from transformers import MambaForCausalLM
from transformers.cache_utils import MambaCache
from transformers.models.mamba.modeling_mamba import MambaCausalLMOutput


class SSRMambaForCausalLM(MambaForCausalLM):
    def __init__(self, config) -> None:
        super().__init__(config)

    def multimodal_embedding(
        self
        , image_embeds: torch.FloatTensor
        , depth_embeds: torch.FloatTensor
        , input_ids: torch.LongTensor
    ) -> torch.FloatTensor:
        input_embeds = []
        for batch_idx, input_id in enumerate(input_ids):
            input_embed, image_cnt, depth_cnt = [], 0, 0
            for token_id in input_id:
                if token_id == self.image_token_id:
                    input_embed.append(image_embeds[batch_idx, image_cnt, :])
                    image_cnt += 1
                elif token_id == self.depth_token_id:
                    input_embed.append(depth_embeds[batch_idx, depth_cnt, :])
                    depth_cnt += 1
                else:
                    input_embed.append(self.get_input_embeddings()(token_id))
            input_embeds.append(torch.stack(input_embed, dim=0))
        return torch.stack(input_embeds, dim=0)
    
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
            inputs_embeds = self.multimodal_embedding(image_embeds, depth_embeds, input_ids)
        mamba_outputs = self.backbone(
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        return mamba_outputs