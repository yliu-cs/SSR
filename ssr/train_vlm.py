import os
import torch
import autoroot
from torch import nn
from ssr.utils.misc import quiet
from accelerate import Accelerator
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "SSR-CoT"))
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser.parse_args()


def train(
    model: nn.Module
    , dataloader: DataLoader
    , optimizer: torch.optim.Optimizer
    , scheduler: torch.optim.lr_scheduler.LRScheduler
    , accelerator: Accelerator
) -> None:
    for batch in dataloader:
        inputs, targets = batch
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


def main(args: Namespace) -> None:
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)
    optimizer, dataloader, scheduler = accelerator.prepare(optimizer, dataloader, scheduler)
    train(model, dataloader, optimizer, scheduler, accelerator)
    accelerator.unwrap_model(model).save_pretrained(
        args.output_dir
        , is_main_process=accelerator.is_main_process
        , save_function=accelerator.save
        , state_dict=accelerator.get_state_dict(model)
    )


if __name__ == "__main__":
    quiet()
    args = get_args()
    main(args)