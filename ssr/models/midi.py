import math
import torch
import autoroot
from torch import nn
from typing import Tuple
from dataclasses import dataclass
from transformers.utils import ModelOutput
from transformers import PretrainedConfig, PreTrainedModel, MambaForCausalLM, Qwen2ForCausalLM


class MIDIConfig(PretrainedConfig):
    model_type = "MIDI"
    def __init__(
        self
        , mamba_path_or_name: str = "state-spaces/mamba-130m-hf"
        , image_dim: int = 1024
        , depth_dim: int = 1152
        , llm_path_or_name: str = "Qwen/Qwen2.5-3B"
        , **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.mamba_path_or_name = mamba_path_or_name
        self.image_dim = image_dim
        self.depth_dim = depth_dim
        self.llm_path_or_name = llm_path_or_name


@dataclass
class MIDIOutput(ModelOutput):
    loss: torch.Tensor = None
    mamba_loss: torch.Tensor = None
    llm_loss: torch.Tensor = None
    tor_embeds: torch.Tensor = None


class MIDI(PreTrainedModel):
    # Mamba-based Image-Depth Interpreter（MIDI）
    def __init__(self, config: MIDIConfig):
        super().__init__(config)
        self.config = config
        self.mamba = MambaForCausalLM.from_pretrained(
            self.config.mamba_path_or_name
            , trust_remote_code=True
        )
        self.image_proj = nn.Sequential(
            nn.Linear(self.config.image_dim, self.mamba.config.hidden_size)
            , nn.GELU()
            , nn.Linear(self.mamba.config.hidden_size, self.mamba.config.hidden_size)
        )
        self.depth_proj = nn.Sequential(
            nn.Linear(self.config.depth_dim, self.mamba.config.hidden_size)
            , nn.GELU()
            , nn.Linear(self.mamba.config.hidden_size, self.mamba.config.hidden_size)
        )
        self.llm = Qwen2ForCausalLM.from_pretrained(
            self.config.llm_path_or_name
            , attn_implementation="flash_attention_2"
            , trust_remote_code=True
        )
        self.tor_proj = nn.Sequential(
            nn.Linear(self.mamba.config.hidden_size, self.llm.config.hidden_size)
            , nn.GELU()
            , nn.Linear(self.llm.config.hidden_size, self.llm.config.hidden_size)
        )
        self.post_init()
    
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(module.bias, -bound, bound)
    
    def forward(
        self
        , mamba_input_ids: torch.LongTensor
        , mamba_attention_mask: torch.LongTensor
        , mamba_labels: torch.LongTensor
        , llm_input_ids: torch.LongTensor
        , llm_attention_mask: torch.LongTensor
        , llm_labels: torch.LongTensor
        , image_embeds: torch.Tensor
        , depth_embeds: torch.Tensor
        , tor_token_id: Tuple[int, int]
        , alignment: bool = False
    ) -> MIDIOutput:
        image_embeds = self.image_proj(image_embeds)
        depth_embeds = self.depth_proj(depth_embeds)
        mamba_input_embeds = self.mamba.get_input_embeddings()(mamba_input_ids)
        mamba_input_embeds = torch.cat([image_embeds, depth_embeds, mamba_input_embeds], dim=1)
        mamba_outputs = self.mamba(
            attention_mask=mamba_attention_mask
            , inputs_embeds=mamba_input_embeds
            , labels=mamba_labels
            , output_hidden_states=True
        )
        mamba_last_hidden_state = mamba_outputs.hidden_states[-1]
        tor_embeds = self.tor_proj(mamba_last_hidden_state[:, image_embeds.size(1) + depth_embeds.size(1):, :][(mamba_input_ids == tor_token_id[0]), :])
        if alignment:
            llm_input_embeds = self.llm.get_input_embeddings()(llm_input_ids)
            llm_input_embeds[(llm_input_ids == tor_token_id[1]), :] = tor_embeds.to(llm_input_embeds.dtype)
            llm_outputs = self.llm(
                inputs_embeds=llm_input_embeds
                , attention_mask=llm_attention_mask
                , labels=llm_labels
            )
            loss = mamba_outputs.loss + llm_outputs.loss
            return MIDIOutput(
                loss=loss
                , mamba_loss=mamba_outputs.loss
                , llm_loss=llm_outputs.loss
                , tor_embeds=tor_embeds
            )
        else:
            return MIDIOutput(tor_embeds=tor_embeds)


if __name__ == "__main__":
    import os
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    from ssr.utils.prompt import SSRSpecialToken
    from ssr.data.ssr_cot import SSRCoTDataset4Reasoning
    from ssr.utils.misc import quiet, str_datetime, count_params
    
    quiet()
    model = MIDI(MIDIConfig(
        mamba_path_or_name="/ssdwork/liuyang/Models/mamba-130m-hf"
        , llm_path_or_name="/ssdwork/liuyang/Models/Qwen2.5-3B"
    )).to(torch.device("cuda"))
    model.eval()
    print(f"{str_datetime()}: {count_params(model)=}")

    mamba_tokenizer = AutoTokenizer.from_pretrained(model.config.mamba_path_or_name)
    mamba_tokenizer.add_tokens(SSRSpecialToken.TOR_TOKEN, special_tokens=True)
    llm_tokenizer = AutoTokenizer.from_pretrained(model.config.llm_path_or_name)
    llm_tokenizer.add_tokens(SSRSpecialToken.TOR_TOKEN, special_tokens=True)
    
    dataset = SSRCoTDataset4Reasoning(
        data_dir=os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "SSR-CoT")
        , n_tor=10
        , mamba_tokenizer=mamba_tokenizer
        , llm_tokenizer=llm_tokenizer
        , max_length=(128, 1024, 128)
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=dataset.collate_fn)
    
    with torch.no_grad():
        for batch in dataloader:
            for k in batch.keys():
                batch[k] = batch[k].to(device=torch.device("cuda"))
            loss, (mamba_loss, llm_loss), tor_embeds = model(
                **batch
                , tor_token_id=(
                    mamba_tokenizer._tokenizer.token_to_id(SSRSpecialToken.TOR_TOKEN)
                    , llm_tokenizer._tokenizer.token_to_id(SSRSpecialToken.TOR_TOKEN)
                )
                , alignment=True
            )
            print(f"{str_datetime()}: {loss.item()=} {mamba_loss.item()=} {llm_loss.item()=} {tor_embeds.size()=}")
            break