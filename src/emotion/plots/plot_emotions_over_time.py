from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

IDENTITY_DIR = Path("/home/moritz/Workspace/masterthesis/data/identities")


def plot_smoothed_emotions_over_time(w_size: int = 5):
    """Plot the emotions over time for each person."""
    df = pd.read_csv(IDENTITY_DIR / "identities.csv")

    grouped = df.groupby("ClassID")

    fig = plt.figure(figsize=(10, 15), tight_layout=True)

    for i, (person_id, group) in enumerate(grouped):
        # Apply a rolling mean to the emotions data with a window size of 10
        emotions_rolling = (
            group[
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
            .rolling(window=w_size)
            .mean()
        )

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


def plot_max_emotions_over_time():
    """Plot the maximum emotion over time for each person."""
    df = pd.read_csv(IDENTITY_DIR / "identities.csv")

    grouped = df.groupby("ClassID")

    fig = plt.figure(figsize=(10, 15), tight_layout=True)

    for i, (person_id, group) in enumerate(grouped):

        # Get the maximum emotion for each frame
        person_max_emotion = group.iloc[:, 7:].idxmax(axis=1)

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
    plot_smoothed_emotions_over_time(150)
    plot_max_emotions_over_time()
