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

from src.emotion.analysis.data_preprocessing import DataPreprocessor, LinearInterpolator

IDENTITY_DIR = Path("/home/moritz/Workspace/masterthesis/data/identities")


def plot_max_emotion_distribution(df: pd.DataFrame):

    emotions = ["Angry", "Disgust", "Happy", "Sad", "Surprise", "Fear", "Neutral"]

    # Find the maximum emotion for each person_id by selecting the column with the maximum value
    df["Max_Emotion"] = df[emotions].idxmax(axis=1)

    grouped = df.groupby("ClassID")
    max_length_group_index = grouped.size().idxmax()
    y_lim = len(grouped.get_group(max_length_group_index))

    fig = plt.figure(figsize=(10, 15), tight_layout=True)

    for i, (person_id, group) in enumerate(grouped):

        # Group the data by person_id and calculate the count of each emotion for each person
        grouped = group.groupby("Max_Emotion").size().reset_index(name="counts")

        ax = fig.add_subplot(2, 2, i + 1)

        # Plot the pivot table as a bar plot
        plt.bar(
            emotions,
            [
                grouped.loc[grouped["Max_Emotion"] == emotion, "counts"].iloc[0]
                if emotion in grouped["Max_Emotion"].values
                else 0
                for emotion in emotions
            ],
        )
        ax.set_title(f"Categorical distr. of maximum emotions for {person_id}")
        ax.set_xlabel("Emotions")
        ax.set_ylabel("Amount")
        ax.set_ylim(0, y_lim)

    # Show the plot
    plt.show()
    fig.savefig(IDENTITY_DIR / "max_emotion_distribution.png")


if __name__ == "__main__":
    df = pd.read_csv(IDENTITY_DIR / "identities.csv")

    preprocessing_pipeline = [LinearInterpolator()]

    preprocessor = DataPreprocessor(preprocessing_pipeline)
    pre_df = preprocessor.preprocess_data(df)

    plot_max_emotion_distribution(pre_df)
