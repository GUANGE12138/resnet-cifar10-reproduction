from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--y_true", required=True, help="txt or csv containing one true label per line")
    parser.add_argument("--y_pred", required=True, help="txt or csv containing one predicted label per line")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    y_true = np.loadtxt(args.y_true, dtype=int)
    y_pred = np.loadtxt(args.y_pred, dtype=int)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(include_values=True, xticks_rotation="vertical")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    print(f"Saved confusion matrix to {out}")


if __name__ == "__main__":
    main()
