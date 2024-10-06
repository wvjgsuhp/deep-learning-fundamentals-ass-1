import numpy as np

from .custom_types import PDFloats


def get_metrics(truth: PDFloats, predictions: PDFloats) -> None:
    print(f"- acc: {np.mean(truth == predictions):.4f}")
