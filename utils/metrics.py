from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class AverageMeter:
    name: str
    total: float = 0.0
    count: int = 0

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n


def accuracy(logits: torch.Tensor, target: torch.Tensor, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        out = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            out.append(correct_k.mul_(100.0 / batch_size))
        return out
