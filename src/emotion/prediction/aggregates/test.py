# import os
# import sys
from pathlib import Path
from typing import Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.emotion.utils.constants import CUSTOM_MODEL_DIR

# parent_folder = os.path.abspath(
#     os.path.join(
#         os.path.dirname(os.path.abspath(__file__)),
#         os.pardir,
#         os.pardir,
#         os.pardir,
#         os.pardir,
#     )
# )
# sys.path.append(parent_folder)


def load_models(folder_path: Path = CUSTOM_MODEL_DIR / "aggregates") -> Dict:

    models = {}
    folder_path = folder_path

    for file_path in folder_path.glob("*"):
        if file_path.is_file():
            model = joblib.load(file_path)

            models[str(file_path.stem)] = model

    return models


def generate_predictions(models, X, y):
    # Generate and return a dictionary of mean absolute error (MAE) scores and prediction arrays for each model
    results = {}
    for model_name, mae_grid_search in models.items():
        # Fit the model
        model = mae_grid_search[0].best_estimator_
        # Make predictions
        y_pred = model.predict(X)
        # Calculate mean squared error and mean absolute error
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        results[model_name] = {"mae": mae, "mse": mse, "y_pred": y_pred}
    return results


def plot_predictions(results, y):
    # Create a figure with subplots for all models
    fig, axes = plt.subplots(
        nrows=len(results), ncols=y.shape[1], figsize=(16, 2 * len(results))
    )

    # Plot actual vs predicted values for each model and target variable
    for i, (model_name, result) in enumerate(results.items()):
        y_pred = result["y_pred"]
        for j in range(y.shape[1]):
            ax = axes[i, j]
            ax.scatter(y[:, j], y_pred[:, j], s=5)
            ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")

            ax.set_ylabel("Predicted", fontsize=8)
            ax.tick_params(axis="both", which="major", labelsize=6)
            if j == 0:
                ax.set_ylabel(f"{model_name}", fontsize=8)
            if i == 0:
                ax.set_title(
                    f"Target variable {j+1}",
                    fontsize=8,
                )
            if i == len(results) - 1:
                ax.set_xlabel("Actual", fontsize=8)

    # Add a common title for the figure
    fig.suptitle("Comparison of model predictions for all target variables")
    fig.subplots_adjust(top=0.9, hspace=0.5, wspace=0.3)
    # fig.tight_layout()
    plt.show()


def plot_mae_scores(results):
    # Create a bar plot of the MAE scores for all models
    fig, ax = plt.subplots(figsize=(8, 6))
    model_names = list(results.keys())
    mae_scores = [results[model_name]["mae"] for model_name in model_names]
    ax.bar(range(len(mae_scores)), mae_scores)
    ax.set_xticks(range(len(mae_scores)))
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title("Comparison of Mean Absolute Error Scores")

    # Add the MAE scores to the plot
    for i, score in enumerate(mae_scores):
        ax.text(i, score, f"{score:.3f}", ha="center", fontsize=8)

    plt.show()


if __name__ == "__main__":
    models = load_models()

    # Define the input and target vectors
    X = np.random.rand(20, 20)
    y = np.random.rand(20, 5)

    # Generate the mean absolute error (MAE) scores and prediction arrays for each model
    results = generate_predictions(models, X, y)

    # Plot the predictions for each model and target variable
    plot_predictions(results, y)

    # Plot the MAE scores for all models
    plot_mae_scores(results)
