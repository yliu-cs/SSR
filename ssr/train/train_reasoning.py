import os
import json
import torch
import autoroot
import numpy as np
from torch import nn
from tqdm.auto import tqdm
from typing import List, Tuple
from accelerate import Accelerator
from torch.utils.data import DataLoader
from ssr.models.midi import MIDIConfig, MIDI
from ssr.utils.prompt import SSRSpecialToken
from argparse import ArgumentParser, Namespace
from ssr.data.ssr_cot import SSRCoTDataset4Reasoning
from ssr.utils.misc import quiet, freeze_module, str_datetime, count_params
from transformers import AutoTokenizer, CLIPProcessor, CLIPVisionModel, SiglipProcessor, SiglipVisionModel, get_cosine_schedule_with_warmup


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Dataset"))
    parser.add_argument("--clip_path", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "clip-vit-large-patch14-336"))
    parser.add_argument("--siglip_path", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "siglip-so400m-patch14-384"))
    parser.add_argument("--n_tor", type=int, default=10)
    parser.add_argument("--max_length", type=Tuple[int, int, int], default=(256, 1024, 256))
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--mamba", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "mamba-130m-hf"))
    parser.add_argument("--llm", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "Qwen2.5-3B"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size_per_gpu", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.getcwd(), "checkpoints", "SSR-Reasoning"))
    return parser.parse_args()


def str_training_state(
    epoch: int
    , epochs: int
    , losses: List[float]
    , mamba_losses: List[float]
    , llm_losses: List[float]
) -> str:
    return " | ".join([
        f"{str_datetime()} [Epoch {epoch + 1}/{epochs}]"
        , f"Loss: {losses[-1]:.4f}"
        , f"Mamba Loss: {mamba_losses[-1]:.4f}"
        , f"LLM Loss: {llm_losses[-1]:.4f}"
    ])


def train(
    model: nn.Module
    , dataloader: DataLoader
    , optimizer: torch.optim.Optimizer
    , scheduler: torch.optim.lr_scheduler.LRScheduler
    , accelerator: Accelerator
    , tor_token_id: Tuple[int, int]
    , epochs: int
) -> Tuple[List[float], List[float], List[float]]:
    losses, mamba_losses, llm_losses = [], [], []
    for epoch in range(epochs):
        progress_bar = tqdm(dataloader, desc=f"{str_datetime()} [Epoch {epoch + 1}/{epochs}]", disable=not accelerator.is_local_main_process)
        for batch in progress_bar:
            outputs = model(**batch, tor_token_id=tor_token_id, alignment=True)
            loss, mamba_loss, llm_loss = [getattr(outputs, key) for key in ("loss", "mamba_loss", "llm_loss")]
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(accelerator.gather(loss).mean().item())
            mamba_losses.append(accelerator.gather(mamba_loss).mean().item())
            llm_losses.append(accelerator.gather(llm_loss).mean().item())
            progress_bar.set_description(" | ".join([str_training_state(epoch, epochs, losses, mamba_losses, llm_losses), f"LR: {scheduler.get_last_lr()[0]:.2e}"]))
    return losses, mamba_losses, llm_losses


def main(args: Namespace) -> None:
    accelerator = Accelerator()
    if accelerator.is_main_process:
        args.output_dir = os.path.join(args.output_dir, str_datetime().strip("[]")[:-4])
        os.makedirs(args.output_dir, exist_ok=True)
        accelerator.print(f"{str_datetime()} {args.output_dir=}")

    accelerator.print(f"{str_datetime()} Loading Tokenizers...")
    mamba_tokenizer = AutoTokenizer.from_pretrained(args.mamba)
    mamba_tokenizer.add_tokens(SSRSpecialToken.TOR_TOKEN, special_tokens=True)
    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm)
    llm_tokenizer.add_tokens(SSRSpecialToken.TOR_TOKEN, special_tokens=True)

    accelerator.print(f"{str_datetime()} Loading CLIP and Siglip Models...")
    clip_processor, clip_model = CLIPProcessor.from_pretrained(args.clip_path), (CLIPVisionModel.from_pretrained(args.clip_path))
    siglip_processor, siglip_model = SiglipProcessor.from_pretrained(args.siglip_path), (SiglipVisionModel.from_pretrained(args.siglip_path))
    clip_model, siglip_model = accelerator.prepare(clip_model), accelerator.prepare(siglip_model)
    accelerator.print(f"{str_datetime()} Loading Dataset...")
    dataset = SSRCoTDataset4Reasoning(
        data_dir=args.data_dir
        , n_tor=args.n_tor
        , mamba_tokenizer=mamba_tokenizer
        , llm_tokenizer=llm_tokenizer
        , max_length=args.max_length
        , clip_processor=clip_processor
        , clip_model=clip_model
        , siglip_processor=siglip_processor
        , siglip_model=siglip_model
    )
    
    tor_token_id = (
        mamba_tokenizer._tokenizer.token_to_id(SSRSpecialToken.TOR_TOKEN)
        , llm_tokenizer._tokenizer.token_to_id(SSRSpecialToken.TOR_TOKEN)
    )
    accelerator.print(f"{str_datetime()} Loading Model...")
    model = MIDI(MIDIConfig(mamba_path_or_name=args.mamba, llm_path_or_name=args.llm))
    
    print('mamba emb is trained? ', model.mamba.get_input_embeddings().weight.requires_grad)
    freeze_module(model.llm)
    accelerator.print(f"{str_datetime()} Model: {count_params(model)}")
    model = accelerator.prepare(model)
    
    accelerator.print(f"{str_datetime()} Preparing Optimizer, Dataloader, Scheduler...")
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)
    dataloader = DataLoader(dataset, batch_size=args.batch_size_per_gpu, shuffle=True, collate_fn=dataset.collate_fn)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer
        , num_warmup_steps=int((len(dataloader) * args.epochs) * args.warmup_ratio)
        , num_training_steps=(len(dataloader) * args.epochs)
    )
    optimizer, dataloader, scheduler = accelerator.prepare(optimizer, dataloader, scheduler)
    accelerator.print(f"{str_datetime()} Training...")
    losses, _, _ = train(model, dataloader, optimizer, scheduler, accelerator, tor_token_id, args.epochs)

    np.save(os.path.join(args.output_dir, "losses.npy"), losses)
    accelerator.print(f"{str_datetime()} Saving Checkpoint into {args.output_dir} ...")
    accelerator.wait_for_everyone()
    accelerator.unwrap_model(model).save_pretrained(
        args.output_dir
        , is_main_process=accelerator.is_main_process
        , save_function=accelerator.save
        , state_dict=accelerator.get_state_dict(model)
    )
    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, "args.json"), "w") as json_file:
            json.dump(vars(args), json_file, indent=4)
    accelerator.print(f"{str_datetime()} Done.")


if __name__ == "__main__":
    quiet()
    args = get_args()
    main(args)
