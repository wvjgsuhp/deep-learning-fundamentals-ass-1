import os
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt


class EDA:
    def __init__(self, df: pd.DataFrame, out_dir: str = "./out") -> None:
        self.df = df
        self.numerical_cols = self.df.select_dtypes([int, float]).columns
        self.out_dir = out_dir

        Path(out_dir).mkdir(parents=True, exist_ok=True)

    def show_missing(self) -> None:
        print("missing values:")
        missings = self.df.isna().sum()
        missings.index = pd.Index([f"- {i}" for i in missings.index])
        print(missings)

        print("zero values:")
        zeros = self.df[self.numerical_cols].eq(0).sum()
        zeros.index = pd.Index([f"- {i}" for i in zeros.index])
        print(zeros)

    def get_histogram(self, x_axis_titles: dict[str, str] = {}) -> None:
        for col in self.numerical_cols:
            ax = self.df[col].hist()
            ax.set_xlabel(x_axis_titles.get(col, col))
            ax.set_ylabel("Count")

            filename = os.path.join(self.out_dir, f"{col}.png")
            plt.savefig(filename)
            plt.close()
