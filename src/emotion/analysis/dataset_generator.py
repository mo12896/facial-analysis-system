# import os
# import sys
from pathlib import Path
from typing import Dict

import networkx as nx
import pandas as pd
from tsfresh.feature_extraction import MinimalFCParameters, extract_features

# grandparent_folder = os.path.abspath(
#     os.path.join(
#         os.path.dirname(os.path.abspath(__file__)),
#         os.pardir,
#         os.pardir,
#         os.pardir,
#     )
# )
# sys.path.append(grandparent_folder)

from src.emotion.analysis.data_preprocessing import (
    DataPreprocessor,
    DerivativesGetter,
    LinearInterpolator,
    RollingAverageSmoother,
    ZeroToOneNormalizer,
)

IDENTITY_DIR = Path("/home/moritz/Workspace/masterthesis/data/identities")


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


# TODO: Note, that we have to stoe the amount of frames into account
# to later weight the vide-clip against all other video-clips per day!
# (Easier: Or just concatemate all identitites.csv files :-))
if __name__ == "__main__":
    emotions = ["Angry", "Disgust", "Happy", "Sad", "Surprise", "Fear", "Neutral"]

    df = pd.read_csv(IDENTITY_DIR / "short_clip_debug.csv")

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
            ],
        }
    ]

    # feature_vectors = time_series_features(pre_df, cols, feature_dict[0])
    # print(feature_vectors)

    # df_counts = max_emotion_features(df, emotions)
    # print(df_counts)

    # presence = presence_features(df)
    # print(presence)

    # gaze_matrix = sna_gaze_features(df)
    # print(gaze_matrix)

    # df_features = pd.concat([feature_vectors, df_counts, presence, gaze_matrix], axis=1)
    # print(df_features)
