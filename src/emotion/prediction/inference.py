from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load

from src.emotion.utils.constants import DATA_DIR_OUTPUT, MODEL_DIR


def preprocess(df: pd.DataFrame, scalers_dict: Dict, selected_features: Dict):
    # Assume you have new data in a DataFrame called `new_data`
    df = df.drop(
        columns=[
            "ClassID",
            "Gazes_StdDev",
            "GazeDifference_StdDev",
            "MutualGaze_StdDev",
        ]
    )
    scaled_data = []
    for feat in df.columns:
        scaler = scalers_dict[feat]
        scaled_feat = scaler.inverse_transform(df[feat].values.reshape(-1, 1)).flatten()
        scaled_data.append(pd.Series(scaled_feat, name=feat))

    # Concatenate scaled features into new DataFrame
    scaled_data = pd.concat(scaled_data, axis=1)

    feature_sets = {}

    for perma_dim, selected_features in selected_features.items():
        # Select the features for the current PERMA dimension
        feature_sets[perma_dim] = selected_features

    return df, feature_sets


def predict(df: pd.DataFrame, features: Dict, path: Path) -> pd.DataFrame:
    best_models = load(path / "best_models.joblib")

    perma = pd.DataFrame(columns=["P", "E", "R", "M", "A"])

    for perma_dim in ["P", "E", "R", "M", "A"]:
        models_path = path / (perma_dim + "/" + best_models[perma_dim] + ".joblib")
        model = load(models_path)
        model = model[0].best_estimator_
        data = df.iloc[:, list(set(features[perma_dim]))]
        result = model.predict(data)
        perma[perma_dim] = result

    return perma


def radar_plot(df: pd.DataFrame):
    data = df.drop(columns=["ClassID"])
    categories = data.columns.tolist()
    num_vars = len(categories)
    num_cols = len(data)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # The plot is a circle, so we need to "complete the loop"
    # and append the start value to the end.
    angles += angles[:1]

    # Create subplots
    fig, axs = plt.subplots(
        ncols=num_cols, figsize=(6 * num_cols, 6), subplot_kw=dict(polar=True)
    )
    if num_cols == 1:  # handle single-column case by putting axs into a list
        axs = [axs]

    for i, (ax, (_, row)) in enumerate(zip(axs, data.iterrows())):
        # Draw one axe per variable + add labels
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Set the x-ticks for the current subplot
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)

        # Draw ylabels
        ax.set_rlabel_position(0)
        ax.yaxis.grid(True)

        # Part 1: Draw the line
        values = row.values.flatten().tolist()
        values += values[:1]  # repeat the first value to close the circular graph
        ax.plot(angles, values, linewidth=1, linestyle="solid", label=df["ClassID"][i])

        # Part 2: Fill area
        ax.fill(angles, values, "b", alpha=0.1)

        # Set the range of y values
        ax.set_ylim(0, 7)

        # Add a legend
        ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

    plt.tight_layout()


def bar_plot(df):
    data = df.drop(columns=["ClassID"])
    # Create a subplot for each row
    num_cols = len(data)
    fig, axs = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5), sharey=True)

    # Handle case with only one row
    if num_cols == 1:
        axs = [axs]

    for row_id, ax in enumerate(axs):
        # Extract the data for the current row
        row_data = data.loc[row_id]
        row_data.plot(kind="bar", ax=ax)
        ax.set_xticklabels(data.columns, rotation=0)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Low", "High"])
        ax.set_title(df["ClassID"][row_id])

    plt.tight_layout()


def perma_inference(prediction: str, dataset: str, filename: str):
    if dataset not in ["small", "big"]:
        raise ValueError("Invalid dataset. Please choose from 'small' or 'big'")

    if prediction not in ["regression", "classification"]:
        raise ValueError(
            "Invalid mode. Please choose from 'regression' or 'classification'"
        )

    path = (
        MODEL_DIR
        / f"custom_models/{'univariate' if prediction == 'regression' else 'classifier'}/{dataset}/"
    )

    # Load the scalers, selected features and dataset
    scalers_dict = load(path / "scalers_dict.joblib")

    selected_features = load(path / "selected_features.joblib")

    df = pd.read_csv(
        DATA_DIR_OUTPUT
        / (
            filename
            + "/extraction_results/"
            + filename
            + "_dataset_"
            + dataset
            + ".csv"
        )
    )

    df_preproc, features = preprocess(df, scalers_dict, selected_features)

    perma = predict(df_preproc, features, path)

    # Save the plot to the defined path
    save_path = DATA_DIR_OUTPUT / ("short_clip/prediction_results/")
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    if prediction == "regression":
        # Assume you have new target data in a DataFrame called `new_target`
        target_scaler = scalers_dict["target_scaler"]
        scaled_perma = pd.DataFrame(
            target_scaler.inverse_transform(perma), columns=["P", "E", "R", "M", "A"]
        )
        scaled_predicts = scaled_perma.clip(lower=0)
        scaled_predicts = pd.concat([df["ClassID"], scaled_predicts], axis=1)

        radar_plot(scaled_predicts)
        plt.savefig(save_path / ("perma_radar_" + dataset + ".png"))

        scaled_predicts.to_csv(
            save_path / ("perma_" + prediction + "_" + dataset + ".csv"), index=False
        )
    else:
        perma = pd.concat([df["ClassID"], perma], axis=1)
        bar_plot(perma)
        plt.savefig(save_path / ("perma_bar_" + dataset + ".png"))

        perma.to_csv(
            save_path / ("perma_" + prediction + "_" + dataset + ".csv"), index=False
        )
