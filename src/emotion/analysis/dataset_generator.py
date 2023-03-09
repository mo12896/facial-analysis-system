import os
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
from tsfresh.feature_extraction import MinimalFCParameters, extract_features

from src.emotion.analysis.data_preprocessing import (
    DataPreprocessor,
    DerivativesGetter,
    LinearInterpolator,
    RollingAverageSmoother,
    ZeroToOneNormalizer,
)

grandparent_folder = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
        os.pardir,
        os.pardir,
    )
)
sys.path.append(grandparent_folder)


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


def presence_features(df: pd.DataFrame) -> pd.Series:
    # Count the number of frames for each ClassID
    class_counts = df["ClassID"].value_counts()
    frames = df["Frame"].nunique()

    # Compute the presence of each ClassID relative to all frames
    presence = class_counts / frames

    return presence


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

    # Count the number of frames for each ClassID
    presence = presence_features(df)
    print(presence)
