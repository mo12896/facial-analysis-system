# import os
# import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

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

from src.emotion.analysis.data_preprocessing import (
    DataPreprocessor,
    LinearInterpolator,
    RollingAverageSmoother,
)

IDENTITY_DIR = Path("/home/moritz/Workspace/masterthesis/data/identities")


def plot_smoothed_emotions_over_time(df: pd.DataFrame):
    """Plot the emotions over time for each person."""

    grouped = df.groupby("ClassID")

    fig = plt.figure(figsize=(10, 15), tight_layout=True)

    for i, (person_id, group) in enumerate(grouped):

        emotions_rolling = group[
            [
                "Frame",
                "Angry",
                "Disgust",
                "Happy",
                "Sad",
                "Surprise",
                "Fear",
                "Neutral",
            ]
        ]

        ax = fig.add_subplot(2, 2, i + 1)
        emotions_rolling.plot(
            x="Frame",
            y=["Angry", "Disgust", "Happy", "Sad", "Surprise", "Fear", "Neutral"],
            ax=ax,
        )

        ax.set_title(f"Emotions over Time for {person_id}")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Confidence")
        ax.set_ylim(0, 1)

    plt.show()
    fig.savefig(IDENTITY_DIR / "emotions_over_time.png")


# TODO: Adapt to new preprocessing pipeline
def plot_max_emotions_over_time(df: pd.DataFrame):
    """Plot the maximum emotion over time for each person."""

    emotions = ["Angry", "Disgust", "Happy", "Sad", "Surprise", "Fear", "Neutral"]

    grouped = df.groupby("ClassID")

    fig = plt.figure(figsize=(10, 15), tight_layout=True)

    for i, (person_id, group) in enumerate(grouped):

        # Get the maximum emotion for each frame
        person_max_emotion = group[emotions].idxmax(axis=1)

        ax = fig.add_subplot(2, 2, i + 1)
        # Plot the maximum emotion over the frame
        ax.plot(group["Frame"], person_max_emotion)
        ax.set_title(f"Maximum Emotions over Time for {person_id}")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Emotion")

    plt.show()
    fig.savefig(IDENTITY_DIR / "max_emotions_over_time.png")


if __name__ == "__main__":
    # If window size is 1, no smoothing is applied
    df = pd.read_csv(IDENTITY_DIR / "identities.csv")

    preprocessing_pipeline = [
        LinearInterpolator(),
        RollingAverageSmoother(
            window_size=150,
            cols=[
                "Angry",
                "Disgust",
                "Happy",
                "Sad",
                "Surprise",
                "Fear",
                "Neutral",
            ],
        ),
    ]

    preprocessor = DataPreprocessor(preprocessing_pipeline)
    pre_df = preprocessor.preprocess_data(df)

    plot_smoothed_emotions_over_time(pre_df)

    # preprocessing_pipeline_2 = [LinearInterpolator()]

    # preprocessor_2 = DataPreprocessor(preprocessing_pipeline_2)
    # pre_df_2 = preprocessor_2.preprocess_data(df)

    # plot_max_emotions_over_time(pre_df_2)
