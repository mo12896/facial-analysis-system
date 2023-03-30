# import os
# import sys
from pathlib import Path
from typing import Dict, Union

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from tsfresh.feature_extraction import (
    EfficientFCParameters,
    MinimalFCParameters,
    extract_features,
)

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
    LinearInterpolator,
    RollingAverageSmoother,
)
from src.emotion.analysis.feature_generator import (
    FeatureGenerator,
    MaxEmotionGenerator,
    VADGenerator,
    VelocityGenerator,
)
from src.emotion.utils.constants import DATA_DIR, IDENTITY_DIR


def compute_custom_ts_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    groups = df.groupby("ClassID")
    features = []

    # Compute the slope, p50, p25, and p75 for each time series in each group
    for name, group in groups:
        x = group["Frame"]
        y = group[col]
        slope, _ = np.polyfit(x, y, 1)
        p25 = y.quantile(0.25)
        p75 = y.quantile(0.75)
        features.append(
            {
                "ClassID": name,
                col + "__Slope": slope,
                col + "__P_25": p25,
                col + "__P_75": p75,
            }
        )

    # Create a dataframe from the features list
    feature_df = pd.DataFrame(features).set_index("ClassID")

    return feature_df


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
        if feature_mode["fc_params"] == MinimalFCParameters():
            extracted_features_cleaned = extracted_features_cleaned.drop(
                extracted_features_cleaned.filter(
                    regex="(" + "|".join(feature_mode["drop"]) + ")"
                ).columns,
                axis=1,
            )
            custom_features = compute_custom_ts_features(df, col)
            extracted_features_cleaned = pd.concat(
                [extracted_features_cleaned, custom_features], axis=1
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


def calculate_difference_matrix(df: pd.DataFrame) -> pd.DataFrame:
    # Compute the difference matrix
    diff_df = df.sub(df.T).abs()

    # Scale the difference matrix to the range 0 to 1 using pandas
    diff_df = (diff_df - diff_df.min(axis=1).min(axis=0)) / (
        diff_df.max(axis=1).max(axis=0) - diff_df.min(axis=1).min(axis=0)
    )
    # diff_df = diff_scaled_df.round(decimals=2)

    return diff_df


def calculate_mutual_gaze_matrix(df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
    # Create a dictionary to store the mutual gaze detections
    mutual_gaze_dict = {}

    # Loop through each frame in the dataframe
    for frame in df["Frame"].unique():

        # Get the rows corresponding to the current frame
        frame_rows = df[df["Frame"] == frame]

        # Get the ClassIDs and GazeDetections for the current frame
        class_ids = list(frame_rows["ClassID"])
        gaze_detections = [
            eval(detections) for detections in frame_rows["GazeDetections"]
        ]

        # Loop through each pair of ClassIDs in the current frame
        for i, class_id_x in enumerate(class_ids):
            for class_id_y in gaze_detections[i]:
                if (class_id_y in class_ids) and (
                    class_id_x in gaze_detections[class_ids.index(class_id_y)]
                ):
                    mutual_gaze_key = (class_id_x, class_id_y)
                    mutual_gaze_dict[mutual_gaze_key] = (
                        mutual_gaze_dict.get(mutual_gaze_key, 0) + 1
                    )

    # Create a pivot table of mutual gazes
    mutual_gaze_matrix = pd.DataFrame(
        mutual_gaze_dict.values(),
        index=pd.MultiIndex.from_tuples(mutual_gaze_dict.keys()),
        columns=["Count"],
    ).unstack(fill_value=0)

    min_val = np.min(mutual_gaze_matrix)
    max_val = np.max(mutual_gaze_matrix)
    mutual_gaze_matrix = (mutual_gaze_matrix - min_val) / (max_val - min_val)

    return mutual_gaze_matrix


def sna_gaze_features(df_gaze: pd.DataFrame) -> pd.DataFrame:

    # Compute degree centrality
    degree_centrality = df_gaze.sum(axis=1)
    degree_centrality_normalized = degree_centrality / df_gaze.to_numpy().sum()

    # Compute in-degree centrality
    in_degree_centrality = df_gaze.sum(axis=0).T
    in_degree_centrality_normalized = in_degree_centrality / df_gaze.to_numpy().sum()

    # Compute betweenness centrality
    G = nx.from_pandas_adjacency(df_gaze)
    betweenness_centrality = pd.Series(nx.betweenness_centrality(G))

    # Create a new dataframe with all the features
    df_features = pd.concat(
        [
            degree_centrality_normalized,
            in_degree_centrality_normalized,
            betweenness_centrality,
        ],
        axis=1,
    )
    df_features.columns = [
        "Degree Centrality",
        "In-degree Centrality",
        "Betweenness Centrality",
    ]

    return df_features


def simple_sna_features(df: pd.DataFrame, feature_name: str) -> pd.DataFrame:

    df = df.replace(0, np.nan)

    # Compute various features for each row
    mean = df.mean(axis=1)
    std_dev = df.std(axis=1)
    min_val = df.min(axis=1)
    max_val = df.max(axis=1)
    range_val = max_val - min_val

    df_features = pd.concat([mean, std_dev, min_val, max_val, range_val], axis=1)
    df_features.columns = [
        feature_name + "_" + feat for feat in ["Mean", "StdDev", "Min", "Max", "Range"]
    ]

    return df_features


def gaze_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df_gaze = create_gaze_matrix(df)

    # Compute the SNA features
    gaze_sna_features = sna_gaze_features(df_gaze)

    # Compute the gaze features
    df_gaze_scaled = (df_gaze - df_gaze.min(axis=1).min(axis=0)) / (
        df_gaze.max(axis=1).max(axis=0) - df_gaze.min(axis=1).min(axis=0)
    )
    gaze_features = simple_sna_features(df_gaze_scaled, feature_name="Gazes")
    # print(gaze_features)

    # Compute the difference features
    diff_df = calculate_difference_matrix(df_gaze)
    diff_features = simple_sna_features(diff_df, feature_name="GazeDifference")
    # print(diff_features)

    # Compute the mutual gaze features
    mutual_df = calculate_mutual_gaze_matrix(df)
    mutual_gaze_features = simple_sna_features(mutual_df, feature_name="MutualGaze")
    # print(mutual_gaze_features)

    # Concatenate the features
    df_features = pd.concat(
        [gaze_sna_features, gaze_features, diff_features, mutual_gaze_features], axis=1
    )
    return df_features


def position_features(
    df: pd.DataFrame, verbose: bool = False, image: bool = False
) -> pd.DataFrame:
    # Rotate the positions for 180 degrees around x-axis
    df["y_center"] = -df["y_center"] + 720

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

        if image:
            # Add an image to the plot
            img = mpimg.imread(str(DATA_DIR / "test_pic.png"))
            ax.imshow(img, extent=[0, 1280, 0, 720], aspect="auto", alpha=0.5)

        sns.set_style("white")
        for class_id, kde in kdes.items():
            sns.kdeplot(
                ax=ax,
                data=df[df["ClassID"] == class_id],
                x="x_center",
                y="y_center",
                cmap="Blues",
                alpha=0.5,
                thresh=0.05,
                fill=True,
            )
            # ax.contour(x, y, kde, levels=5, colors="k", linewidths=0.5)
            ax.text(
                x=df[df["ClassID"] == class_id]["x_center"].mean(),
                y=df[df["ClassID"] == class_id]["y_center"].mean(),
                s=class_id,
                fontsize=16,
            )

        fig.suptitle(
            "Kernel Density Estimation for Positional Occupation of Different ClassIDs",
            fontsize=20,
        )
        ax.set_xlabel("X Center")
        ax.set_ylabel("Y Center")
        ax.set_xlim(0, 1280)
        ax.set_ylim(0, 720)

        plt.show()

    return stds


def process(
    df: pd.DataFrame, ts_feature_dict: dict, path: Path, save: bool = True
) -> None:

    emotions = ["Angry", "Disgust", "Happy", "Sad", "Surprise", "Fear", "Neutral"]
    vad = ["Valence", "Arousal", "Dominance"]

    # Generate features
    feature_pipeline = [MaxEmotionGenerator(), VADGenerator(), VelocityGenerator()]

    feature_generator = FeatureGenerator(feature_pipeline)
    pre_df = feature_generator.generate_features(df)

    # Preprocess data
    preprocessing_pipeline = [
        LinearInterpolator(),
        RollingAverageSmoother(window_size=5, cols=["Velocity"]),
        RollingAverageSmoother(
            window_size=150,
            cols=emotions,
        ),
    ]

    preprocessor = DataPreprocessor(preprocessing_pipeline)
    pre_df = preprocessor.preprocess_data(pre_df)
    pre_df["Velocity"].fillna(0, inplace=True)

    cols = [*emotions, *vad, "Brightness", "Velocity"]
    ts_feature_vectors = time_series_features(pre_df, cols, ts_feature_dict)
    print(ts_feature_vectors)

    max_emotions = max_emotion_features(pre_df, emotions)
    print(max_emotions)

    presence = presence_features(df)
    print(presence)

    # gaze_matrix = sna_gaze_features(df)
    gaze_features = gaze_feature_pipeline(df)
    print(gaze_features)

    pos_features = position_features(pre_df)
    print(pos_features)

    df_features = pd.concat(
        [ts_feature_vectors, max_emotions, presence, gaze_features, pos_features],
        axis=1,
    )

    if save:
        # save the dataframe to a CSV file
        filename = str(path).split(".")[0] + "_dataset_small.csv"
        df_features = df_features.reset_index().rename(columns={"index": "ClassID"})
        df_features.to_csv(filename, index=False)

    print(df_features)


if __name__ == "__main__":
    save: bool = True

    ts_feature_dict = [
        {
            "name": "MinimalFCParameters",
            "fc_params": MinimalFCParameters(),
            "drop": [
                "__sum_values",
                "__length",
                "__absolute_maximum",
                "__variance",
                "__root_mean_square",
            ],
        },
        {
            "name": "EfficientFCParameters",
            "fc_params": EfficientFCParameters(),
            "drop": [],
        },
    ]

    # Load the identity file

    teams = [
        "team_01",
        "team_02",
        "team_03",
        "team_04",
        "team_05",
        "team_06",
        "team_07",
        "team_08",
        "team_09",
        "team_10",
        "team_11",
        "team_12",
        "team_13",
        "team_15",
        "team_16",
        "team_17",
        "team_18",
        "team_19",
        "team_20",
        "team_22",
    ]

    days = ["2023-01-10", "2023-01-12", "2023-01-13"]

    for team in teams:
        for day in days:
            filename = team + "_" + day + ".csv"
            path = IDENTITY_DIR / team / day / filename
            df = pd.read_csv(path)

            process(df, ts_feature_dict[0], path=path, save=save)
            print("Finished processing: " + filename)
