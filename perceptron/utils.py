import numpy as np
from sklearn.metrics import roc_auc_score

from .custom_types import PDFloats


def get_metrics(truth: PDFloats, predictions: PDFloats) -> None:
    print(f"- acc: {np.mean(truth == predictions):.4f}")
    print(f"- roc: {roc_auc_score(truth, predictions):.4f}")
