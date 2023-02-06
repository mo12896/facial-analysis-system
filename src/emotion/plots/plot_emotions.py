from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

IDENTITY_DIR = Path("/home/moritz/Workspace/masterthesis/data/identities")


def plot_emotions_over_time():
    """Plot the emotions over time for each person."""
    df = pd.read_csv(IDENTITY_DIR / "identities.csv")

    person_ids = df["ClassID"].unique()

    _, axs = plt.subplots(2, 2, figsize=(10, 15), tight_layout=True)
    axs = axs.ravel()

    for i, person_id in enumerate(person_ids):
        person_df = df[df["ClassID"] == person_id]
        person_df.plot(
            x="Frame",
            y=["Angry", "Disgust", "Happy", "Sad", "Surprise", "Fear", "Neutral"],
            ax=axs[i],
        )
        axs[i].set_title(f"Emotions for person_id: {person_id}")
        axs[i].set_xlabel("Frame")
        axs[i].set_ylabel("Confidence")

    plt.show()


def plot_max_emotions_over_time():
    """Plot the maximum emotion over time for each person."""
    df = pd.read_csv(IDENTITY_DIR / "identities.csv")

    person_ids = df["ClassID"].unique()

    _, axs = plt.subplots(2, 2, figsize=(10, 15), tight_layout=True)
    axs = axs.ravel()

    for i, person_id in enumerate(person_ids):
        person = df[df["ClassID"] == person_id]

        # Get the maximum emotion for each frame
        person_max_emotion = person.iloc[:, 7:].idxmax(axis=1)

        # Plot the maximum emotion over the frame
        axs[i].plot(person["Frame"], person_max_emotion)
        axs[i].set_title(f"Maximum Emotion for {person_id}")
        axs[i].set_xlabel("Frame")
        axs[i].set_ylabel("Emotion")
    plt.show()


if __name__ == "__main__":
    plot_emotions_over_time()
    plot_max_emotions_over_time()
