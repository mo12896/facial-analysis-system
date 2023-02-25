# import os
# import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from src.emotion.analysis.data_preprocessing import (
    DataPreprocessor,
    DerivativesGetter,
    LinearInterpolator,
    RollingAverageSmoother,
    ZeroToOneNormalizer,
)

# grandparent_folder = os.path.abspath(
#     os.path.join(
#         os.path.dirname(os.path.abspath(__file__)),
#         os.pardir,
#         os.pardir,
#         os.pardir,
#         os.pardir,
#     )
# )
# sys.path.append(grandparent_folder)


IDENTITY_DIR = Path("/home/moritz/Workspace/masterthesis/data/identities")


def plot_smoothed_motion_over_time(df: pd.DataFrame, plot: bool = True) -> Figure:

    # group the data by ClassID and Frame
    grouped = df.groupby("ClassID")
    max_height = df["Derivatives"].max()

    fig = plt.figure(figsize=(20, 5), tight_layout=True)
    fig.suptitle("Motion over Time")

    for i, (person_id, group) in enumerate(grouped):

        smoothed_derivatives = group["Derivatives"]
        # x = smoothed_derivatives.index.values
        x = range(len(smoothed_derivatives))

        ax = fig.add_subplot(1, 4, i + 1)
        # plot the smoothed derivatives over time
        plt.plot(x, smoothed_derivatives.values)
        ax.set_ylim(0, max_height)
        ax.set_title(f"Center derivatives w.r.t. time for {person_id}")
        ax.set_xlabel("Frames")
        ax.set_ylabel("Derivatives")

    if plot:
        plt.show()

    fig.savefig(IDENTITY_DIR / "point_derivatives_over_time.png")

    return fig


if __name__ == "__main__":
    # read the csv file into a pandas DataFrame
    df = pd.read_csv(IDENTITY_DIR / "identities.csv")

    preprocessing_pipeline = [
        LinearInterpolator(),
        DerivativesGetter(),
        RollingAverageSmoother(window_size=5, cols=["Derivatives"]),
        ZeroToOneNormalizer(cols=["Derivatives"]),
    ]

    preprocessor = DataPreprocessor(preprocessing_pipeline)
    pre_df = preprocessor.preprocess_data(df)

    # If window size is 1, no smoothing is applied
    plot_smoothed_motion_over_time(pre_df)
