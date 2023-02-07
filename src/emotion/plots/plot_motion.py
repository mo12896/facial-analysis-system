import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

IDENTITY_DIR = Path("/home/moritz/Workspace/masterthesis/data/identities")


def compute_gradients(point_list: list[int]):
    gradients = []

    for i in range(len(point_list) - 1):
        if i == 0:
            gradient = point_list[i + 1] - point_list[i]
            gradients.append(gradient)
        else:
            gradient = (
                abs(point_list[i + 1] - point_list[i])
                - abs(point_list[i - 1] - point_list[i])
            ) / 2
            gradients.append(gradient)

    return gradients


def compute_center_point(min, max):
    center = 0.5 * (float(min) + float(max))
    return center


def estimate_point_density(x: list, y: list):
    # Kernel density estimation
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    return z


def compute_statistics(x: list, y: list, person_id: str):
    # Convert the points to a Pandas DataFrame
    points = list(zip(x, y))
    df = pd.DataFrame(points, columns=["x", "y"])

    stats = df.describe()

    return stats


def plot_point_derivatives():
    # Read the CSV file into a list of dictionaries
    rows = []
    with open(IDENTITY_DIR / "identities.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    person_ids = ["person_id1", "person_id2", "person_id3", "person_id4"]

    _, axs = plt.subplots(2, 2, figsize=(10, 15), tight_layout=True)
    axs = axs.ravel()

    for i, person_id in enumerate(person_ids):

        # Filter the data to get only the rows for person_id1
        person_data = [row for row in rows if row["ClassID"] == person_id]

        # Compute the center point (x, y) of the bounding box for each frame
        x, y = [], []
        for row in person_data:
            x.append(compute_center_point(row["XMin"], row["XMax"]))
            y.append(compute_center_point(row["YMin"], row["YMax"]))

        # Compute the derivative of the center point (x, y) over different frames
        dx = compute_gradients(x)
        dy = compute_gradients(y)

        stats = compute_statistics(dx, dy, person_id)
        x_mean = stats.loc["mean", "x"]
        y_mean = stats.loc["mean", "y"]

        # Estimate the point density
        z = estimate_point_density(dx, dy)

        # Plot the derivative of the center point (x, y)
        axs[i].scatter(dx, dy, c=z, s=100)
        axs[i].scatter(x_mean, y_mean, c="red", marker="x", s=50)

        size = 20
        axs[i].set_xlim(left=-size, right=size)
        axs[i].set_ylim(bottom=-size, top=size)
        axs[i].grid(True)

        axs[i].set_xlabel("dx")
        axs[i].set_ylabel("dy")
        axs[i].set_title(f"Center point derivatives for {person_id}")

    plt.show()


def plot_point_positions():
    # Read the CSV file into a list of dictionaries
    rows = []
    with open(IDENTITY_DIR / "identities.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    person_ids = ["person_id1", "person_id2", "person_id3", "person_id4"]

    _, axs = plt.subplots(2, 2, figsize=(10, 15), tight_layout=True)
    axs = axs.ravel()

    for i, person_id in enumerate(person_ids):

        # Filter the data to get only the rows for person_id1
        person_data = [row for row in rows if row["ClassID"] == person_id]

        # Compute the center point (x, y) of the bounding box for each frame
        # TODO: points are flipped!
        x, y = [], []
        for row in person_data:
            x.append(compute_center_point(row["XMin"], row["XMax"]))
            y.append(compute_center_point(row["YMin"], row["YMax"]))

        # Plot the derivative of the center point (x, y)
        axs[i].scatter(x, y, s=5)
        axs[i].set_xlim(left=0, right=1848)
        axs[i].set_ylim(bottom=0, top=1053)
        axs[i].set_xlabel("x")
        axs[i].set_ylabel("y")
        axs[i].set_title(f"Center point for {person_id} over different frames")
    plt.show()


if __name__ == "__main__":
    plot_point_derivatives()
