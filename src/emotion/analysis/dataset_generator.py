# import os
# import sys
from typing import Dict

# import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from tsfresh.feature_extraction import MinimalFCParameters, extract_features

from src.emotion.analysis.data_preprocessing import (
    DataPreprocessor,
    DerivativesGetter,
    LinearInterpolator,
    RollingAverageSmoother,
    ZeroToOneNormalizer,
)
from src.emotion.utils.constants import IDENTITY_DIR

# grandparent_folder = os.path.abspath(
#     os.path.join(
#         os.path.dirname(os.path.abspath(__file__)),
#         os.pardir,
#         os.pardir,
#         os.pardir,
#     )
# )
# sys.path.append(grandparent_folder)


def time_series_features(
    df: pd.DataFrame, cols: list[str], feature_mode: Dict
) -> pd.DataFrame:
    feature_df = []

    for col in cols:
        extracted_features = extract_features(
            df,
            column_id="ClassID",
            column_value=col,
            default_fc_parameters=feature_mode["fc_params"],
        )
        extracted_features_cleaned = extracted_features.dropna(axis=1, how="all")
        extracted_features_cleaned = extracted_features_cleaned.drop(
            extracted_features_cleaned.filter(
                regex="(" + "|".join(feature_mode["drop"]) + ")"
            ).columns,
            axis=1,
        )
        feature_df.append(extracted_features_cleaned)

    feature_df = pd.concat(feature_df, axis=1)

    return feature_df


def max_emotion_features(df: pd.DataFrame, emotions: list[str]) -> pd.DataFrame:
    def count_max_emotion_changes(group):
        changes = sum(
            group["Max_Emotion"].iloc[i] != group["Max_Emotion"].iloc[i - 1]
            for i in range(1, len(group))
        )
        return changes / len(group)

    # Create max emotion features
    df["Max_Emotion"] = df[emotions].idxmax(axis=1)
    emotions_max_count = [emotion + "__max_count" for emotion in emotions]
    df_counts = pd.DataFrame(columns=emotions_max_count + ["Freq_Emotion_Changes"])
    grouped = df.groupby("ClassID")

    for person_id, group in grouped:
        counts = group["Max_Emotion"].value_counts(normalize=True)
        max_emotion_changes = count_max_emotion_changes(group)
        row = [counts.get(emotion, 0) for emotion in emotions] + [max_emotion_changes]
        df_counts.loc[person_id] = row

    return df_counts


def presence_features(df: pd.DataFrame) -> pd.DataFrame:
    class_counts = df["ClassID"].value_counts()
    frames = df["Frame"].nunique()

    # Compute the presence of each ClassID relative to all frames
    presence = class_counts / frames

    feature_df = pd.DataFrame(columns=["Presence"])
    feature_df["Presence"] = presence

    return feature_df


def create_gaze_matrix(df: pd.DataFrame) -> pd.DataFrame:
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

    # Create a list of the ClassIDs in the same order as the rows and columns of the matrix
    class_ids = sorted(list(counts.keys()))

    # Create an empty matrix
    matrix = [[0 for _ in range(len(class_ids))] for _ in range(len(class_ids))]

    # Fill in the matrix
    for i, class_id1 in enumerate(class_ids):
        for j, class_id2 in enumerate(class_ids):
            if class_id1 in counts and class_id2 in counts[class_id1]:
                matrix[j][i] = counts[class_id1][class_id2]

    # Create a DataFrame from the matrix and assign the ClassIDs to the row and column indices
    df_gaze = pd.DataFrame(matrix, index=class_ids, columns=class_ids)

    return df_gaze


def sna_gaze_features(df: pd.DataFrame) -> pd.DataFrame:
    df_gaze = create_gaze_matrix(df)

    # Compute degree centrality
    degree_centrality = df_gaze.sum(axis=1)
    degree_centrality_normalized = degree_centrality / df_gaze.to_numpy().sum()

    # Compute in-degree centrality
    in_degree_centrality = df_gaze.sum(axis=0).T
    in_degree_centrality_normalized = in_degree_centrality / df_gaze.to_numpy().sum()

    # Compute betweenness centrality using NetworkX
    G = nx.from_pandas_adjacency(df_gaze)
    betweenness_centrality = pd.Series(nx.betweenness_centrality(G))

    # Compute mutual gaze
    mutual_gaze = df_gaze / df_gaze.sum().sum()

    # Create a new dataframe with all the features
    df_features = pd.concat(
        [
            degree_centrality_normalized,
            in_degree_centrality_normalized,
            betweenness_centrality,
            mutual_gaze,
        ],
        axis=1,
    )
    df_features.columns = [
        "Degree Centrality",
        "In-degree Centrality",
        "Betweenness Centrality",
        "Mutual Gaze_p1",
        "Mutual Gaze_p2",
        "Mutual Gaze_p3",
        "Mutual Gaze_p4",
    ]

    return df_features


def position_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    # Rotate the positions for 180 degrees around x-axis
    df["y_center"] = df["y_center"].apply(lambda y: -y) + 720

    stds = pd.DataFrame(columns=["Std_X_Center", "Std_Y_Center"])
    kdes = {}
    # Compute kernel density estimation for each ClassID
    for class_id in df["ClassID"].unique():
        # Select the data for the current ClassID
        data = df[df["ClassID"] == class_id][["x_center", "y_center"]]

        # Compute the kernel density estimation
        k = gaussian_kde(data.T)

        # Compute the standard deviation of the KDE in each direction
        x_std = np.sqrt(k.covariance[0, 0])
        y_std = np.sqrt(k.covariance[1, 1])

        stds.loc[class_id] = [x_std, y_std]

        if verbose:
            x, y = np.mgrid[
                data["x_center"].min() : data["x_center"].max() : 100j,
                data["y_center"].min() : data["y_center"].max() : 100j,
            ]

            positions = np.vstack([x.ravel(), y.ravel()])
            z = np.reshape(k(positions).T, x.shape)

            # Store the KDE in a dictionary
            kdes[class_id] = z

    if verbose:
        # Plot the KDEs in a single plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Add an image to the plot
        # img = mpimg.imread(str(DATA_DIR / "test_pic.png"))
        # ax.imshow(img, extent=[0, 1280, 0, 720], aspect="auto", alpha=0.5)

        sns.set_style("white")
        for class_id, kde in kdes.items():
            sns.kdeplot(
                ax=ax,
                data=df[df["ClassID"] == class_id],
                x="x_center",
                y="y_center",
                cmap="Blues",
                alpha=0.5,
                shade=True,
                thresh=0.05,
                fill=True,
            )
            ax.contour(x, y, kde, levels=5, colors="k", linewidths=0.5)
            ax.text(
                x=df[df["ClassID"] == class_id]["x_center"].mean(),
                y=df[df["ClassID"] == class_id]["y_center"].mean(),
                s=class_id,
                fontsize=16,
            )

        fig.suptitle("Kernel Density Estimation for Different ClassIDs", fontsize=20)
        ax.set_xlabel("X Center")
        ax.set_ylabel("Y Center")
        ax.set_xlim(0, 1280)
        ax.set_ylim(0, 720)

        plt.show()

    return stds


# TODO: Note, that we have to stoe the amount of frames into account
# to later weight the vide-clip against all other video-clips per day!
# (Easier: Or just concatemate all identitites.csv files :-))
if __name__ == "__main__":
    emotions = ["Angry", "Disgust", "Happy", "Sad", "Surprise", "Fear", "Neutral"]

    filename = "short_clip_debug.csv"
    df = pd.read_csv(IDENTITY_DIR / filename)

    preprocessing_pipeline = [
        LinearInterpolator(),
        DerivativesGetter(),
        RollingAverageSmoother(window_size=5, cols=["Derivatives"]),
        ZeroToOneNormalizer(cols=["Derivatives", "Brightness"]),
        RollingAverageSmoother(
            window_size=150,
            cols=emotions,
        ),
    ]

    preprocessor = DataPreprocessor(preprocessing_pipeline)
    pre_df = preprocessor.preprocess_data(df)

    # Create time series feature
    cols = [*emotions, "Brightness", "Derivatives"]
    feature_dict = [
        {
            "name": "MinimalFCParameters",
            "fc_params": MinimalFCParameters(),
            "drop": [
                "__sum_values",
                "__length",
                "__maximum",
                "__absolute_maximum",
                "__minimum",
                "__median",
                "__variance",
            ],
        }
    ]

    feature_vectors = time_series_features(pre_df, cols, feature_dict[0])
    print(feature_vectors)

    df_counts = max_emotion_features(df, emotions)
    print(df_counts)

    presence = presence_features(df)
    print(presence)

    gaze_matrix = sna_gaze_features(df)
    print(gaze_matrix)

    pos_feature = position_features(df, verbose=False)
    print(pos_feature)

    df_features = pd.concat(
        [feature_vectors, df_counts, presence, gaze_matrix, pos_feature], axis=1
    )
    # save the dataframe to a CSV file
    dataset = filename.split(".")[0] + "_dataset.csv"
    df_features.to_csv(str(IDENTITY_DIR / dataset), index=False)

    print(df_features)
