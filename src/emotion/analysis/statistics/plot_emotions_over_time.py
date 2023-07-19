# import os
# import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from src.emotion.analysis.data_preprocessing import (
    DataPreprocessor,
    LinearInterpolator,
    RollingAverageSmoother,
)
from src.emotion.analysis.feature_generator import FeatureGenerator, VADGenerator
from src.emotion.utils.constants import DATA_DIR_OUTPUT

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

emotions = ["Angry", "Disgust", "Happy", "Sad", "Surprise", "Fear", "Neutral"]
vad = ["Valence", "Arousal", "Dominance"]


def plot_smoothed_emotions_over_time(
    df: pd.DataFrame, filename: str, plot: bool = True
) -> Figure:
    """Plot the emotions over time for each person."""

    grouped = df.groupby("ClassID")

    fig = plt.figure(figsize=(20, 5), tight_layout=True)
    fig.suptitle("Emotions over Time")

    for i, (person_id, group) in enumerate(grouped):
        emotions_rolling = group[["Frame", *emotions]]

        ax = fig.add_subplot(1, 4, i + 1)
        emotions_rolling.plot(
            x="Frame",
            y=emotions,
            ax=ax,
        )

        ax.set_title(f"Emotions over Time for {person_id}")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Confidence")
        ax.set_ylim(0, 1)

    if plot:
        plt.show()
    path = DATA_DIR_OUTPUT / (filename + "/extraction_results/")
    fig.savefig(path / "emotions_over_time.png")

    return fig


def plot_smoothed_vad_values(
    df: pd.DataFrame, filename: str, emo_window: int, plot: bool = True
) -> Figure:
    # Create a dictionary with the given data
    mapping = {
        "Angry": [-0.43, 0.67, 0.34],
        "Happy": [0.76, 0.48, 0.35],
        "Surprise": [0.4, 0.67, -0.13],
        "Disgust": [-0.6, 0.35, 0.11],
        "Fear": [-0.64, 0.6, -0.43],
        "Sad": [-0.63, 0.27, -0.33],
        "Neutral": [0.0, 0.0, 0.0],
    }

    features_pipeline = [VADGenerator(mapping=mapping)]

    feature_generator = FeatureGenerator(features_pipeline)
    feature_df = feature_generator.generate_features(df)

    preprocessing_pipeline = [
        LinearInterpolator(),
        RollingAverageSmoother(
            window_size=emo_window,
            cols=[*emotions, *vad],
        ),
    ]

    preprocessor = DataPreprocessor(preprocessing_pipeline)
    pre_df = preprocessor.preprocess_data(feature_df)

    grouped = pre_df.groupby("ClassID")

    fig = plt.figure(figsize=(20, 5), tight_layout=True)
    fig.suptitle("VAD-values over Time")

    for i, (person_id, group) in enumerate(grouped):
        emotions_rolling = group[["Frame", *vad]]

        ax = fig.add_subplot(1, 4, i + 1)
        emotions_rolling.plot(
            x="Frame",
            y=vad,
            ax=ax,
        )

        ax.set_title(f"VAD-values over Time for {person_id}")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Confidence")
        ax.set_ylim(-1, 1)

    if plot:
        plt.show()
    path = DATA_DIR_OUTPUT / (filename + "/extraction_results/")
    fig.savefig(path / "vad_over_time.png")

    return fig


# TODO: Adapt to new preprocessing pipeline
def plot_max_emotions_over_time(
    df: pd.DataFrame, filename: str, plot: bool = True
) -> Figure:
    """Plot the maximum emotion over time for each person."""

    emotions = ["Angry", "Disgust", "Happy", "Sad", "Surprise", "Fear", "Neutral"]

    grouped = df.groupby("ClassID")

    fig = plt.figure(figsize=(20, 5), tight_layout=True)
    fig.suptitle("Maximum Emotions over Time")

    for i, (person_id, group) in enumerate(grouped):
        # Get the maximum emotion for each frame
        person_max_emotion = group[emotions].idxmax(axis=1)

        ax = fig.add_subplot(1, 4, i + 1)
        # Plot the maximum emotion over the frame
        ax.plot(group["Frame"], person_max_emotion)
        ax.set_title(f"Maximum Emotions over Time for {person_id}")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Emotion")

    if plot:
        plt.show()
    path = DATA_DIR_OUTPUT / (filename + "/extraction_results/")
    fig.savefig(path / "max_emotions_over_time.png")

    return fig


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
