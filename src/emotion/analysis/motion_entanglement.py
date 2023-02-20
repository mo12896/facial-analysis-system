from pathlib import Path

import numpy as np
import pandas as pd
from data_preprocessing import (
    DataPreprocessor,
    LinearInterpolator,
    RangeZeroToOneNormalizer,
    RollingAverageSmoother,
)
from entanglement import SimilarityMetric, plot_entanglement_graph
from plot_utils import plot_time_series

IDENTITY_DIR = Path("/home/moritz/Workspace/masterthesis/data/identities")


class DerivativesGetter:
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:

        grouped = df.groupby("ClassID")

        for _, group in grouped:
            group_devs = abs(
                group["x_center"].diff() / group["y_center"].diff()
            ).fillna(0)
            df.loc[group.index, "Derivatives"] = group_devs

        return df


def plot_motion_entanglement(
    df: pd.DataFrame,
    window_size: int = 5,
    metric: SimilarityMetric = SimilarityMetric.euclidean,
    save_fig: bool = False,
) -> None:

    preprocessing_steps = [
        LinearInterpolator(),
        DerivativesGetter(),
        RollingAverageSmoother(window_size=window_size, cols=["Derivatives"]),
        RangeZeroToOneNormalizer(cols=["Derivatives"]),
    ]

    preprocessor = DataPreprocessor(preprocessing_steps)
    pre_df = preprocessor.preprocess_data(df)

    # Plot the emotion entanglement
    grouped = pre_df.groupby("ClassID")["Derivatives"].apply(np.array).to_numpy()
    X = np.vstack(grouped)
    plot_time_series(X, df["ClassID"].unique())

    # Upscale for nice plot:
    X *= 50

    # Plot the motion entanglement
    plot_entanglement_graph(X, metric, pre_df["ClassID"].unique(), "Motion", save_fig)


if __name__ == "__main__":
    # Load the data
    df = pd.read_csv(IDENTITY_DIR / "identities.csv")

    plot_motion_entanglement(df, 5, SimilarityMetric.euclidean)
