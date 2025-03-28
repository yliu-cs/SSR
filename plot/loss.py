import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.ndimage import gaussian_filter1d
from argparse import Namespace, ArgumentParser


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default=os.path.join(os.getcwd(), "checkpoints"))
    parser.add_argument("--figure_dir", type=str, default=os.path.join(os.getcwd(), "figure"))
    parser.add_argument("--stage", type=str, default="Reasoning", choices=["Reasoning", "VLM"])
    parser.add_argument("--pdf", action="store_true", help="Export PDF")
    args = parser.parse_args()
    return args


def main(args: Namespace):
    os.makedirs(args.figure_dir, exist_ok=True)

    colors = [
        ("#E1F2FC", "#5AA0F7")
        , ("#B1C5FD", "#7B87FF")
        , ("#FBB3E5", "#E95A85")
        , ("#C4B4E5", "#A373C8")
    ]

    plt.rc("font", **{"family": "Times New Roman", "size": 12})
    fig, ax = plt.subplots()
    losses = []
    versions = list(filter(lambda x: os.path.exists(os.path.join(args.checkpoint_dir, f"SSR-{args.stage}", x, "losses.npy")), os.listdir(os.path.join(args.checkpoint_dir, f"SSR-{args.stage}"))))
    if args.stage == "Reasoning":
        versions = sorted(
            versions
            , key=lambda x: (
                os.path.basename(json.load(open(os.path.join(args.checkpoint_dir, f"SSR-{args.stage}", x, "args.json")))["llm"])
                , os.path.basename(json.load(open(os.path.join(args.checkpoint_dir, f"SSR-{args.stage}", x, "args.json")))["mamba"])
            )
        )
    elif args.stage == "VLM":
        versions = sorted(
            versions
            , key=lambda x: (
                os.path.basename(json.load(open(os.path.join(json.load(open(os.path.join(args.checkpoint_dir, f"SSR-{args.stage}", x, "args.json")))["pretrained_midi"], "args.json")))["mamba"])
                , os.path.basename(json.load(open(os.path.join(args.checkpoint_dir, f"SSR-{args.stage}", x, "args.json")))["pretrained_vlm"])
            )
        )
    for i, version in enumerate(versions):
        loss_path = os.path.join(args.checkpoint_dir, f"SSR-{args.stage}", version, "losses.npy")
        if not os.path.exists(loss_path):
            continue
        loss = np.load(loss_path).tolist()
        version_cfg = json.load(open(os.path.join(args.checkpoint_dir, f"SSR-{args.stage}", version, "args.json")))
        if args.stage == "Reasoning":
            mamba, llm = "Mamba-" + os.path.basename(version_cfg["mamba"]).split("-")[1].upper(), os.path.basename(version_cfg["llm"])
        else:
            mamba = "Mamba-" + os.path.basename(json.load(open(os.path.join(version_cfg["pretrained_midi"], "args.json")))["mamba"]).split("-")[1].upper()
            vlm = "-".join(os.path.basename(version_cfg["pretrained_vlm"]).split("-")[:-1])
        ax.plot(range(len(loss)), loss, color=colors[i][0], linestyle="-", alpha=0.5, zorder=1)
        smoothed_loss = gaussian_filter1d(loss, sigma=300)
        ax.plot(range(len(smoothed_loss)), smoothed_loss, color=colors[i][1], linestyle="-", linewidth=3 if args.stage == "Reasoning" else 1, label=f"{mamba} & {llm if args.stage == 'Reasoning' else vlm}", zorder=100)
        losses += loss
    ax.set_xlim(left=0, right=len(losses) // len(versions))
    ax.set_ylim(bottom=min(losses) - 0.1, top=sorted(losses)[math.ceil(len(losses) * (0.9985 if args.stage == "VLM" else 0.99))])
    ax.xaxis.set_major_locator(plt.MultipleLocator(10000))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: int(x // 10000)))
    ax.set_xlabel("Steps (1e4)")
    ax.set_ylabel("Loss")
    for spine in ["top", "right"]:
        ax.spines[spine].set_color("none")
    legend = ax.legend(loc="upper right", markerscale=0.5, handlelength=1, prop={"size": 10})
    for line in legend.get_lines():
        line.set_linewidth(3)
    ax.grid(
        axis="y"
        , linestyle=(0, (5, 10))
        , linewidth=0.25
        , color="#4E616C"
        , zorder=-100
    )
    if args.pdf:
        plt.savefig(os.path.join(args.figure_dir, f"{args.stage}_Loss.pdf"))
        if os.path.exists(os.path.join(args.figure_dir, f"{args.stage}_Loss.png")):
            os.remove(os.path.join(args.figure_dir, f"{args.stage}_Loss.png"))
    else:
        plt.savefig(os.path.join(args.figure_dir, f"{args.stage}_Loss.png"), dpi=600)
    plt.close()


if __name__ == "__main__":
    args = get_args()
    main(args)