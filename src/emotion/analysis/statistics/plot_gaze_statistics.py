from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.emotion.utils.constants import DATA_DIR_OUTPUT

IDENTITY_DIR = Path("/home/moritz/Workspace/masterthesis/data/identities")


def plot_gaze_statistics(df: pd.DataFrame, filename: str, plot: bool = True) -> Figure:
    # Create a dictionary to hold the counts
    counts = {}

    # Loop over the rows in the DataFrame
    for _, row in df.iterrows():
        # Get the GazeDetections
        detections = row["GazeDetections"]
        # Skip rows with empty GazeDetections
        if detections == "[]":
            continue
        # Get the ClassID
        class_id = row["ClassID"]
        # If the ClassID is not in the counts dictionary, add it
        if class_id not in counts:
            counts[class_id] = {}
        # Loop over the other ClassIDs in the GazeDetections
        for other_id in eval(detections):
            # If the other ClassID is not in the counts dictionary, add it
            if other_id not in counts:
                counts[other_id] = {}
            # If the ClassID is not already in the other ClassID's counts, add it
            if class_id not in counts[other_id]:
                counts[other_id][class_id] = 0
            # Increment the count
            counts[other_id][class_id] += 1

    # Create a list of the ClassIDs in the same order as the rows of the matrix
    class_ids = sorted(list(counts.keys()))

    # Create an empty matrix
    matrix = [[0 for _ in range(len(class_ids))] for _ in range(len(class_ids))]

    # Fill in the matrix
    for i, class_id1 in enumerate(class_ids):
        for j, class_id2 in enumerate(class_ids):
            if class_id1 in counts and class_id2 in counts[class_id1]:
                matrix[j][i] = counts[class_id1][class_id2]

    # Plot the matrix
    fig, ax = plt.subplots()
    fig.suptitle("Gaze Detections Matrix")
    ax.imshow(matrix, cmap="Blues")

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(class_ids)))
    ax.set_yticks(np.arange(len(class_ids)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(class_ids)
    ax.set_yticklabels(class_ids)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(class_ids)):
        for j in range(len(class_ids)):
            ax.text(j, i, matrix[i][j], ha="center", va="center", color="w")

    fig.tight_layout()

    if plot:
        plt.show()

    path = DATA_DIR_OUTPUT / (filename + "/extraction_results/")
    fig.savefig(path / "gaze_matrix.png")

    return fig


if __name__ == "__main__":
    df = pd.read_csv(IDENTITY_DIR / "identities.csv")

    plot_gaze_statistics(df, str(IDENTITY_DIR))
