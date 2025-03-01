import torch
from einops import rearrange
from dataclasses import dataclass
from transformers.cache_utils import Cache
from typing import List, Tuple, Optional, Union
from transformers.modeling_outputs import ModelOutput
from .modeling_internlm3 import Cache, Unpack, KwargsForCausalLM, InternLM3ForCausalLM


@dataclass
class SSRCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    tor_embeds: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class SSRInternlm3ForCausalLM(InternLM3ForCausalLM):
    def __init__(self, config) -> None:
        super().__init__(config)
    
    def multimodal_embedding(
        self
        , image_embeds: torch.FloatTensor
        , depth_embeds: torch.FloatTensor
        , tor_embeds: torch.FloatTensor
        , input_ids: torch.LongTensor
    ) -> torch.FloatTensor:
        input_embeds = []
        for batch_idx, input_id in enumerate(input_ids):
            input_embed, image_cnt, depth_cnt, tor_cnt = [], 0, 0, 0
            for token_id in input_id:
                if token_id == self.image_token_id:
                    input_embed.append(image_embeds[batch_idx, image_cnt, :])
                    image_cnt += 1
                elif token_id == self.depth_token_id:
                    input_embed.append(depth_embeds[batch_idx, depth_cnt, :])
                    depth_cnt += 1
                elif token_id == self.tor_token_id:
                    input_embed.append(tor_embeds[batch_idx, tor_cnt, :])
                    tor_cnt += 1
                else:
                    input_embed.append(self.get_input_embeddings()(token_id))
            input_embeds.append(torch.stack(input_embed, dim=0))
        return torch.stack(input_embeds, dim=0)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        image_embeds: torch.FloatTensor = None,
        depth_embeds: torch.FloatTensor = None,
        tor_embeds: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, SSRCausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if inputs_embeds is None:
            inputs_embeds = self.multimodal_embedding(image_embeds, depth_embeds, tor_embeds, input_ids)
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
        last_hidden_state = outputs.last_hidden_state
        logits = self.lm_head(last_hidden_state)
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        tor_embeds = last_hidden_state[(input_ids == self.tor_token_id), :]
        tor_embeds = rearrange(tor_embeds, f"(b l) d -> b l d", b=last_hidden_state.size(0))
        return SSRCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            tor_embeds=tor_embeds,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )