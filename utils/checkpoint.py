from __future__ import annotations

from pathlib import Path
import torch


def save_checkpoint(path: str | Path, model, optimizer, scheduler, epoch: int, best_metric: float, config: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "best_metric": best_metric,
            "config": config,
        },
        path,
    )


def load_checkpoint(path: str | Path, model, map_location="cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    return ckpt
