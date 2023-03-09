# import os
# import sys
from pathlib import Path
from typing import Dict

import pandas as pd

# from tsfresh import extract_features,
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


def compute_time_series_features(
    df: pd.DataFrame, features: list[str], feature_mode: Dict
) -> pd.DataFrame:
    feature_df = []

    for feature in features:
        extracted_features = extract_features(
            df,
            column_id="ClassID",
            column_value=feature,
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
    pre_df["Max_Emotion"] = pre_df[emotions].idxmax(axis=1)

    features = [*emotions, "Brightness", "Derivatives"]
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

    feature_df = compute_time_series_features(pre_df, features, feature_dict[0])

    print(feature_df)
