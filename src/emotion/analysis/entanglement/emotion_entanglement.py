from pathlib import Path

import numpy as np
import pandas as pd
from data_preprocessing import (
    DataPreprocessor,
    LinearInterpolator,
    RollingAverageSmoother,
)

from src.emotion.analysis.entanglement import SimilarityMetric, plot_entanglement_graph
from src.emotion.analysis.plot_utils import plot_time_series

IDENTITY_DIR = Path("/home/moritz/Workspace/masterthesis/data/identities")

# X = np.apply_along_axis(
#     lambda row: np.convolve(row, np.ones(150) / 150, mode="same"),
#     axis=1,
#     arr=X,
# )


def plot_emotional_entanglement(
    df: pd.DataFrame,
    emotion: list[str],
    window_size: int = 150,
    metric: SimilarityMetric = SimilarityMetric.euclidean,
    save_fig: bool = False,
) -> None:

    preprocessing_pipeline = [
        LinearInterpolator(),
        RollingAverageSmoother(window_size=window_size, cols=emotion),
    ]

    preprocessor = DataPreprocessor(preprocessing_pipeline)
    pre_df = preprocessor.preprocess_data(df)

    grouped = pre_df.groupby("ClassID")[emotion[0]].apply(np.array).to_numpy()
    X = np.nan_to_num(np.vstack(grouped))

    plot_time_series(X, pre_df["ClassID"].unique())
    # Use dtw, correlation or euclidean
    plot_entanglement_graph(X, metric, pre_df["ClassID"].unique(), emotion[0], save_fig)


if __name__ == "__main__":
    df = pd.read_csv(IDENTITY_DIR / "identities.csv")

    # Plot the emotion entanglement
    plot_emotional_entanglement(df, ["Surprise"], 150, SimilarityMetric.euclidean)
