import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers.cache_utils import Cache
from typing import List, Tuple, Optional, Union
from transformers.modeling_outputs import ModelOutput
from .modeling_internlm3 import Cache, Unpack, KwargsForCausalLM, InternLM3PreTrainedModel, InternLM3Model


@dataclass
class SSRCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    tor_embeds: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class SSRForCausalLM(InternLM3PreTrainedModel):
    _auto_class = "AutoModelForCausalLM"
    _tied_weights_keys = ["output.weight"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.model = InternLM3Model(config)
        self.vocab_size = config.vocab_size
        self.output = nn.Linear(config.hidden_size, config.vocab_size - 2, bias=False)
        self.max_length = config.max_length
        self.post_init()
        # TODO: Image Component
        self.image_encoder = None
        self.image_proj = None
        # TODO: Depth Component
        self.depth_encoder = None
        self.depth_proj = None

    def merge_input_embeds_with_image_depth(
        self
        , image_embeds: torch.FloatTensor
        , depth_embeds: torch.FloatTensor
        , tor_embeds: torch.FloatTensor
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
        # Merge Image Tor Embeds
        if tor_embeds is not None and input_ids.size(1) != 1:
            batch_idx_tor_embeds = 0
            for batch_idx, input_id in enumerate(input_ids):
                matching = torch.where(input_id  == self.config.image_tor_token_id)
                num_tor_tokens_per_sample = len(matching[0])
                inputs_embeds[batch_idx][matching] = tor_embeds[batch_idx_tor_embeds:batch_idx_tor_embeds + num_tor_tokens_per_sample].to(inputs_embeds.dtype)
                batch_idx_tor_embeds += num_tor_tokens_per_sample
    
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
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, SSRCausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            self.merge_input_embeds_with_image_depth(image_embeds, depth_embeds, tor_embeds, inputs_embeds, input_ids)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.output(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return SSRCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            tor_embeds=hidden_states[torch.where(input_ids == self.config.tor_token_index)],
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )