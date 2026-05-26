from __future__ import annotations

import argparse
import torch
import torch.nn as nn

from datasets import build_dataloaders
from models import build_model
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from train import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "cuda") == "cuda" else "cpu")
    _, _, test_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"test_loss={test_loss:.4f}, test_acc={test_acc:.2f}")


if __name__ == "__main__":
    main()
