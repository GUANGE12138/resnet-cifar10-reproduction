from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from datasets import build_dataloaders
from models import build_model
from utils.checkpoint import save_checkpoint
from utils.config import load_config
from utils.logger import CSVLogger
from utils.metrics import AverageMeter, accuracy
from utils.seed import set_seed


def build_optimizer(model: nn.Module, cfg: dict):
    tr = cfg.get("training", {})
    name = tr.get("optimizer", "sgd").lower()
    lr = float(tr.get("lr", 0.1))
    wd = float(tr.get("weight_decay", 1e-4))
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=float(tr.get("momentum", 0.9)), weight_decay=wd)
    if name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, cfg: dict):
    tr = cfg.get("training", {})
    name = tr.get("scheduler", "multistep").lower()
    if name == "multistep":
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=tr.get("milestones", [60, 90]),
            gamma=float(tr.get("gamma", 0.1)),
        )
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(tr.get("epochs", 100)))
    if name in {"none", "null"}:
        return None
    raise ValueError(f"Unknown scheduler: {name}")


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, amp=True):
    model.train()
    losses = AverageMeter("loss")
    accs = AverageMeter("acc")
    for data, target in tqdm(loader, desc="train", leave=False):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp and device.type == "cuda"):
            logits = model(data)
            loss = criterion(logits, target)
        if scaler is not None and amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        batch_size = data.size(0)
        losses.update(loss.item(), batch_size)
        accs.update(accuracy(logits, target, topk=(1,))[0].item(), batch_size)
    return losses.avg, accs.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses = AverageMeter("loss")
    accs = AverageMeter("acc")
    for data, target in tqdm(loader, desc="eval", leave=False):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).long()
        logits = model(data)
        loss = criterion(logits, target)
        batch_size = data.size(0)
        losses.update(loss.item(), batch_size)
        accs.update(accuracy(logits, target, topk=(1,))[0].item(), batch_size)
    return losses.avg, accs.avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to yaml config.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    exp = cfg.get("experiment_name", Path(args.config).stem)
    out_dir = Path(cfg.get("logging", {}).get("output_dir", "outputs"))
    log_path = out_dir / "logs" / f"{exp}.csv"
    ckpt_path = out_dir / "checkpoints" / f"{exp}_best.pt"

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "cuda") == "cuda" else "cpu")
    train_loader, val_loader, _ = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = GradScaler(enabled=bool(cfg.get("training", {}).get("amp", True)) and device.type == "cuda")
    logger = CSVLogger(log_path, ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc"])

    best_acc = -1.0
    epochs = int(cfg.get("training", {}).get("epochs", 100))
    for epoch in range(1, epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler=scaler, amp=bool(cfg.get("training", {}).get("amp", True))
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()
        logger.log({
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
        print(f"[{exp}] epoch={epoch:03d} lr={lr:.5g} train_acc={train_acc:.2f} val_acc={val_acc:.2f}")
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, best_acc, cfg)
    logger.close()
    print(f"Best val_acc={best_acc:.2f}; checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
