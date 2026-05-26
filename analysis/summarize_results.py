from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = []
    for path, label in zip(args.logs, args.labels):
        df = pd.read_csv(path)
        best_idx = df["val_acc"].idxmax()
        rows.append({
            "model": label,
            "best_epoch": int(df.loc[best_idx, "epoch"]),
            "best_val_acc": float(df.loc[best_idx, "val_acc"]),
            "final_train_acc": float(df.iloc[-1]["train_acc"]),
            "final_val_acc": float(df.iloc[-1]["val_acc"]),
        })
    out_df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(out_df.to_string(index=False))
    print(f"Saved summary to {out}")


if __name__ == "__main__":
    main()
