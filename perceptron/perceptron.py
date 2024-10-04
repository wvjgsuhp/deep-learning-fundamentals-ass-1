import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .custom_types import NPFloatMatrix, NPFloats, PDFloats


class Perceptron:
    def fit(
        self,
        x: NPFloatMatrix | pd.DataFrame,
        y: NPFloats | PDFloats,
        learning_rate: float,
        epochs: int,
        early_stopping: int = 0,
        x_validate: NPFloatMatrix | pd.DataFrame | None = None,
        y_validate: NPFloats | PDFloats | None = None,
        random_seed: int = 0,
    ) -> None:
        if early_stopping > 0:
            if x_validate is None or y_validate is None:
                raise ValueError("`x_validate` and `y_validate` must be provided when `early_stopping` > 0")

            if isinstance(y_validate, pd.Series):
                y_validate = y_validate.to_numpy()

            x_validate = self._add_bias(x_validate)

            no_improvement_steps = 0
            self.best_weights = np.array([])
            best_auc = 0

        if isinstance(y, pd.Series):
            y = y.to_numpy()

        x = self._add_bias(x)
        rng = np.random.default_rng(random_seed)
        self.weights = rng.random(x.shape[1])

        for epoch in range(epochs):
            predictions = self._predict(x)
            loss = predictions - y
            self.weights -= learning_rate * (x.T * loss).sum(axis=1)

            if early_stopping > 0:
                predictions_validate = self._predict(x_validate)
                auc = roc_auc_score(y_validate, predictions_validate)
                if auc > best_auc:
                    best_auc = auc
                    self.best_weights = self.weights.copy()
                    no_improvement_steps = 0
                else:
                    no_improvement_steps += 1

                if no_improvement_steps >= early_stopping:
                    print(f"Stopping after {epoch} epochs: auc={auc:.4f}")
                    break

    def predict(self, x: NPFloatMatrix | pd.DataFrame) -> NPFloats:
        x = self._add_bias(x)
        return self._predict(x)

    def _add_bias(self, x: NPFloatMatrix | pd.DataFrame) -> NPFloatMatrix:
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()

        return np.append(np.ones((x.shape[0], 1)), x, axis=1)

    def _predict(self, x: NPFloatMatrix | pd.DataFrame) -> NPFloats:
        return self._activate(x @ self.weights.reshape(-1, 1)).flatten()

    def _activate(self, x: NPFloats) -> NPFloats:
        threshold = 0
        return np.where(x >= threshold, 1, 0)
