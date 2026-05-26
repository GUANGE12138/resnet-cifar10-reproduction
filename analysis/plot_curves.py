from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--metric", default="val_acc", choices=["train_acc", "val_acc", "train_loss", "val_loss"])
    args = parser.parse_args()

    if len(args.logs) != len(args.labels):
        raise ValueError("--logs and --labels must have the same length.")

    plt.figure(figsize=(8, 5))
    for path, label in zip(args.logs, args.labels):
        df = pd.read_csv(path)
        plt.plot(df["epoch"], df[args.metric], label=label)
    plt.xlabel("Epoch")
    plt.ylabel(args.metric)
    plt.legend()
    plt.grid(True, alpha=0.3)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    print(f"Saved figure to {out}")


if __name__ == "__main__":
    main()
