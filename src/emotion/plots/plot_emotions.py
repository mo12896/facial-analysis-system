from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

IDENTITY_DIR = Path("/home/moritz/Workspace/masterthesis/data/identities")


def plot_emotions_over_time():
    """Plot the emotions over time for each person."""
    df = pd.read_csv(IDENTITY_DIR / "identities.csv")

    person_ids = df["ClassID"].unique()

    fig = plt.figure(figsize=(10, 15), tight_layout=True)

    for i, person_id in enumerate(person_ids):
        person_df = df[df["ClassID"] == person_id]
        ax = fig.add_subplot(2, 2, i + 1)
        person_df.plot(
            x="Frame",
            y=["Angry", "Disgust", "Happy", "Sad", "Surprise", "Fear", "Neutral"],
            ax=ax,
        )

        ax.set_title(f"Emotions for person_id: {person_id}")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Confidence")

    plt.show()
    fig.savefig(IDENTITY_DIR / "emotions_over_time.png")


def plot_max_emotions_over_time():
    """Plot the maximum emotion over time for each person."""
    df = pd.read_csv(IDENTITY_DIR / "identities.csv")

    person_ids = df["ClassID"].unique()

    fig = plt.figure(figsize=(10, 15), tight_layout=True)

    for i, person_id in enumerate(person_ids):
        person = df[df["ClassID"] == person_id]

        # Get the maximum emotion for each frame
        person_max_emotion = person.iloc[:, 7:].idxmax(axis=1)

        ax = fig.add_subplot(2, 2, i + 1)
        # Plot the maximum emotion over the frame
        ax.plot(person["Frame"], person_max_emotion)
        ax.set_title(f"Maximum Emotion for {person_id}")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Emotion")

    plt.show()
    fig.savefig(IDENTITY_DIR / "max_emotions_over_time.png")


if __name__ == "__main__":
    plot_emotions_over_time()
    plot_max_emotions_over_time()
