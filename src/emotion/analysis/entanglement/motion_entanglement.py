# import os
# import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.emotion.analysis.data_preprocessing import (
    DataPreprocessor,
    DerivativesGetter,
    LinearInterpolator,
    RollingAverageSmoother,
    ZeroToOneNormalizer,
)
from src.emotion.analysis.entanglement import SimilarityMetric, plot_entanglement_graph
from src.emotion.analysis.plot_utils import plot_time_series

# grandparent_folder = os.path.abspath(
#     os.path.join(
#         os.path.dirname(os.path.abspath(__file__)),
#         os.pardir,
#         os.pardir,
#         os.pardir,
#     )
# )
# sys.path.append(grandparent_folder)


IDENTITY_DIR = Path("/home/moritz/Workspace/masterthesis/data/identities")


def plot_motion_entanglement(
    df: pd.DataFrame,
    window_size: int = 5,
    metric: SimilarityMetric = SimilarityMetric.euclidean,
    save_fig: bool = False,
) -> None:

    preprocessing_pipeline = [
        LinearInterpolator(),
        DerivativesGetter(),
        RollingAverageSmoother(window_size=window_size, cols=["Derivatives"]),
        ZeroToOneNormalizer(cols=["Derivatives"]),
    ]

    preprocessor = DataPreprocessor(preprocessing_pipeline)
    pre_df = preprocessor.preprocess_data(df)

    # Plot the emotion entanglement
    grouped = pre_df.groupby("ClassID")["Derivatives"].apply(np.array).to_numpy()
    X = np.vstack(grouped)

    max_height = pre_df["Derivatives"].max()
    plot_time_series(X, df["ClassID"].unique(), max_height)

    # Upscale for nice plot:
    X *= 25

    # Plot the motion entanglement
    plot_entanglement_graph(X, metric, pre_df["ClassID"].unique(), "Motion", save_fig)


if __name__ == "__main__":
    # Load the data
    df = pd.read_csv(IDENTITY_DIR / "identities.csv")

    plot_motion_entanglement(df, 5, SimilarityMetric.euclidean)
