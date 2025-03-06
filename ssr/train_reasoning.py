import os
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
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from ssr.utils.misc import quiet, freeze_module, str_datetime, accelerate_print, count_params


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "SSR-CoT"))
    parser.add_argument("--n_tor", type=int, default=10)
    parser.add_argument("--max_length", type=Tuple[int, int, int], default=(128, 1024, 128))
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mamba", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "mamba-130m-hf"))
    parser.add_argument("--llm", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "Qwen2.5-3B"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size_per_gpu", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.01)
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
) -> None:
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
    os.makedirs(args.output_dir, exist_ok=True)
    accelerator = Accelerator()

    accelerate_print(f"{str_datetime()} Loading Tokenizers...", accelerator.is_main_process)
    mamba_tokenizer = AutoTokenizer.from_pretrained(args.mamba)
    mamba_tokenizer.add_tokens(SSRSpecialToken.TOR_TOKEN, special_tokens=True)
    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm)
    llm_tokenizer.add_tokens(SSRSpecialToken.TOR_TOKEN, special_tokens=True)

    accelerate_print(f"{str_datetime()} Loading Dataset...", accelerator.is_main_process)
    dataset = SSRCoTDataset4Reasoning(
        data_dir=args.data_dir
        , n_tor=args.n_tor
        , mamba_tokenizer=mamba_tokenizer
        , llm_tokenizer=llm_tokenizer
        , max_length=args.max_length
    )

    accelerate_print(f"{str_datetime()} Loading Model...", accelerator.is_main_process)
    model = MIDI(MIDIConfig(mamba_path_or_name=args.mamba, llm_path_or_name=args.llm))
    freeze_module(model.llm)
    accelerate_print(f"{str_datetime()} Model: {count_params(model)}", accelerator.is_main_process)

    accelerate_print(f"{str_datetime()} Preparing Optimizer, Dataloader, Scheduler...", accelerator.is_main_process)
    model = accelerator.prepare(model)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)
    dataloader = DataLoader(dataset, batch_size=args.batch_size_per_gpu, shuffle=True, collate_fn=dataset.collate_fn)
    num_training_steps = len(dataloader) * args.epochs // accelerator.num_processes
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer
        , num_warmup_steps=int(num_training_steps * args.warmup_ratio)
        , num_training_steps=num_training_steps
    )
    optimizer, dataloader, scheduler = accelerator.prepare(optimizer, dataloader, scheduler)
    tor_token_id = (
        mamba_tokenizer._tokenizer.token_to_id(SSRSpecialToken.TOR_TOKEN)
        , llm_tokenizer._tokenizer.token_to_id(SSRSpecialToken.TOR_TOKEN)
    )
    accelerate_print(f"{str_datetime()} Training...", accelerator.is_main_process)
    losses, _, _ = train(model, dataloader, optimizer, scheduler, accelerator, tor_token_id, args.epochs)

    np.save(os.path.join(args.output_dir, "losses.npy"), losses)
    accelerate_print(f"{str_datetime()} Saving Checkpoint...", accelerator.is_main_process)
    accelerator.wait_for_everyone()
    accelerator.unwrap_model(model).save_pretrained(
        args.output_dir
        , is_main_process=accelerator.is_main_process
        , save_function=accelerator.save
        , state_dict=accelerator.get_state_dict(model)
    )
    accelerate_print(f"{str_datetime()} Done.", accelerator.is_main_process)


if __name__ == "__main__":
    quiet()
    args = get_args()
    main(args)