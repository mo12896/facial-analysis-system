from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

IDENTITY_DIR = Path("/home/moritz/Workspace/masterthesis/data/identities")


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Read the maximum emotions for each person."""

    # create an empty DataFrame to hold the interpolated data
    interpolated_df = pd.DataFrame(columns=df.columns)

    # group the DataFrame by ClassID
    grouped = df.groupby("ClassID")

    max_length = grouped.apply(lambda x: len(x)).max()

    # iterate over each group and interpolate missing frames
    for _, group in grouped:
        # create a new DataFrame to hold the interpolated frames for this ClassID
        # Iterate over the filtered data and fill in missing frames
        last_frame = None
        new_rows = []
        group_length = group.apply(lambda x: len(x)).max()
        for i, (_, row) in enumerate(group.iterrows()):
            if last_frame is None:
                last_frame = row
                new_rows.append(row)
            elif i + 1 == group_length:
                diff = max_length - row["Frame"]
                if diff:
                    for j in range(diff):
                        new_row = row.copy()
                        new_row["Frame"] = row["Frame"] + j + 1
                        new_rows.append(new_row)
            else:
                while last_frame["Frame"] < row["Frame"] - 1:
                    missing_frame = last_frame["Frame"] + 1
                    interp_row = last_frame.copy()
                    interp_row["Frame"] = missing_frame
                    for col in df.columns:
                        if col not in ["Frame", "ClassID"]:
                            interp_row[col] = (
                                last_frame[col] * (row["Frame"] - missing_frame)
                                + row[col] * (missing_frame - last_frame["Frame"])
                            ) / (row["Frame"] - last_frame["Frame"])
                    new_rows.append(interp_row)
                    last_frame = interp_row
                new_rows.append(row)
                last_frame = row

        # Combine the new rows with the original data and sort by frame

        interpolated_df = pd.concat(
            [interpolated_df, pd.DataFrame(new_rows)], ignore_index=True
        )
        interpolated_df = interpolated_df.sort_values(by=["ClassID", "Frame"])

    return interpolated_df


def max_emotions(df: pd.DataFrame) -> np.ndarray:

    emotions = ["Angry", "Disgust", "Happy", "Sad", "Surprise", "Fear", "Neutral"]

    # create a new column called "Max-Emotion" that contains the maximum emotion value for each row
    df["Max-Emotion"] = df[emotions].max(axis=1).tolist()

    # group the dataframe by "ClassID" and store the maximum emotion value for each group in a separate row
    grouped = (
        df.groupby("ClassID")
        .apply(lambda group: group[emotions].max(axis=1).tolist())
        .apply(np.array)
        .to_numpy()
    )
    grouped = np.vstack(grouped)

    return grouped


def single_emotion(df: pd.DataFrame, emotion: str, window: int = 200) -> np.ndarray:
    grouped = df.groupby("ClassID")[emotion].apply(np.array).to_numpy()
    X = np.vstack(grouped)
    X = np.apply_along_axis(
        lambda row: np.convolve(row, np.ones(window) / window, mode="valid"),
        axis=1,
        arr=X,
    )

    return X


def plot_emotions(X: np.ndarray, labels) -> None:
    # to set the plot size
    plt.figure(figsize=(16, 8), dpi=150)

    # using plot method to plot open prices.
    # in plot method we set the label and color of the curve.
    plt.plot(X[0, :], label=labels[0])
    plt.plot(X[1, :], label=labels[1])
    plt.plot(X[2, :], label=labels[2])
    plt.plot(X[3, :], label=labels[3])

    # adding Label to the x-axis
    plt.xlabel("Frames")
    plt.legend()
    plt.show()


def plot_emotion_entanglement(X: np.ndarray, labels) -> None:

    # Compute pairwise distances between all nodes using Euclidean distance
    distances = pdist(X)

    # Convert condensed distance matrix to square distance matrix
    dist_matrix = squareform(distances)

    # Create a graph from the distance matrix
    G = nx.from_numpy_array(dist_matrix)

    # Draw the graph with node positions determined by multidimensional scaling
    pos = nx.drawing.layout.spring_layout(G, dim=2)

    nx.draw(G, pos)

    # Add labels to the nodes
    labels = {i: class_id for i, class_id in enumerate(labels)}
    nx.draw_networkx_labels(G, pos, labels)

    # Show the plot
    plt.show()


def plot_emotion_entanglement_debug(df: pd.DataFrame, emotion: str, window: int = 200):
    groups = df.groupby("ClassID")
    ids = df["ClassID"].unique()

    grouped = groups[emotion].apply(np.array).to_numpy()
    X = np.vstack(grouped)
    X = np.apply_along_axis(
        lambda row: np.convolve(row, np.ones(window) / window, mode="valid"),
        axis=1,
        arr=X,
    )
    plot_emotions(X, ids)

    # Compute pairwise distances between all nodes using Euclidean distance
    distances = pdist(X)

    # Convert condensed distance matrix to square distance matrix
    dist_matrix = squareform(distances)

    # Create a graph from the distance matrix
    G = nx.from_numpy_array(dist_matrix)

    # TODO: Here is a bug!!!
    # Draw the graph with node positions determined by multidimensional scaling
    pos = nx.drawing.layout.spring_layout(G, dim=2)
    print(pos)

    nx.draw(G, pos)

    # Add labels to the nodes
    labels = {i: class_id for i, class_id in enumerate(ids)}
    nx.draw_networkx_labels(G, pos, labels)

    # Show the plot
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv(IDENTITY_DIR / "identities.csv")
    pre_df = preprocess_data(df)
    # max_emotions = single_emotion(pre_df, "Surprise", window=150)
    # plot_emotions(max_emotions, df["ClassID"].unique())
    # plot_emotion_entanglement(max_emotions, df["ClassID"].unique())
    plot_emotion_entanglement_debug(pre_df, "Happy", window=150)
