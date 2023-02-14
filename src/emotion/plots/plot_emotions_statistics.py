from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

IDENTITY_DIR = Path("/home/moritz/Workspace/masterthesis/data/identities")


def plot_max_emotion_distribution():
    # Load the csv file into a pandas dataframe
    df = pd.read_csv(IDENTITY_DIR / "identities.csv")

    # Find the maximum emotion for each person_id by selecting the column with the maximum value
    df["Max_Emotion"] = df[
        ["Angry", "Disgust", "Happy", "Sad", "Surprise", "Fear", "Neutral"]
    ].idxmax(axis=1)

    grouped = df.groupby("ClassID")

    emotions = ["Angry", "Disgust", "Happy", "Sad", "Surprise", "Fear", "Neutral"]

    fig = plt.figure(figsize=(10, 15), tight_layout=True)

    for i, (person_id, group) in enumerate(grouped):

        # Group the data by person_id and calculate the count of each emotion for each person
        grouped = group.groupby("Max_Emotion").size().reset_index(name="counts")

        ax = fig.add_subplot(2, 2, i + 1)

        # Plot the pivot table as a bar plot
        # plt.bar(grouped["Max_Emotion"], grouped["counts"])
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

    # Show the plot
    plt.show()
    fig.savefig(IDENTITY_DIR / "max_emotion_distribution.png")


if __name__ == "__main__":
    plot_max_emotion_distribution()
