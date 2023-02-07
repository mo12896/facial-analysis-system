import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

IDENTITY_DIR = Path("/home/moritz/Workspace/masterthesis/data/identities")


# TODO: Check what the best way for computing the gradients is!
def compute_gradients(point_list: list[int]):
    gradients = []

    for i in range(len(point_list) - 1):

        gradient = point_list[i + 1] - point_list[i]
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


def perpare_data(x, y):
    # Define the borders
    deltaX = (max(x) - min(x)) / 10
    deltaY = (max(y) - min(y)) / 10
    xmin = min(x) - deltaX
    xmax = max(x) + deltaX
    ymin = min(y) - deltaY
    ymax = max(y) + deltaY
    print(xmin, xmax, ymin, ymax)
    # Create meshgrid
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    # Kernel density estimation
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    return xx, yy, f, xmin, xmax, ymin, ymax


def plot_point_derivatives():
    # Read the CSV file into a list of dictionaries
    rows = []
    with open(IDENTITY_DIR / "identities.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    person_ids = ["person_id1", "person_id2", "person_id3", "person_id4"]

    fig = plt.figure(figsize=(10, 15), tight_layout=True)

    for i, person_id in enumerate(person_ids):

        # Filter the data to get only the rows for person_id1
        person_data = [row for row in rows if row["ClassID"] == person_id]

        # Compute the center point (x, y) of the bounding box for each frame
        x, y = [], []
        for row in person_data:
            x.append(compute_center_point(row["XMin"], row["XMax"]))
            y.append(compute_center_point(row["YMin"], row["YMax"]))

        # Compute the gradients of the point
        dx, dy = compute_gradients(x), compute_gradients(y)
        gradients = [x / y if y != 0 else 0 for x, y in zip(dx, dy)]

        z = gaussian_kde(np.array(gradients))

        # Plot the gradients
        ax = fig.add_subplot(2, 2, i + 1)
        x_range = np.linspace(min(gradients), max(gradients), 100)
        plt.plot(x_range, z(x_range))
        size = 30
        ax.set_xlim(left=-size, right=size)
        ax.set_title(f"Point Gradients for {person_id}")
        ax.set_xlabel("Gradient")
        ax.set_ylabel("PDF")

    plt.show()
    fig.savefig(IDENTITY_DIR / "point_derivatives.png")


def plot_2d_point_derivatives():
    # Read the CSV file into a list of dictionaries
    rows = []
    with open(IDENTITY_DIR / "identities.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    person_ids = ["person_id1", "person_id2", "person_id3", "person_id4"]

    fig = plt.figure(figsize=(10, 15), tight_layout=True)

    for i, person_id in enumerate(person_ids):

        # Filter the data to get only the rows for person_id1
        person_data = [row for row in rows if row["ClassID"] == person_id]

        # Compute the center point (x, y) of the bounding box for each frame
        x, y = [], []
        for row in person_data:
            x.append(compute_center_point(row["XMin"], row["XMax"]))
            y.append(compute_center_point(row["YMin"], row["YMax"]))

        # Compute the derivative of the center point (x, y) over different frames
        dx, dy = compute_gradients(x), compute_gradients(y)

        stats = compute_statistics(dx, dy, person_id)
        x_mean = stats.loc["mean", "x"]
        y_mean = stats.loc["mean", "y"]

        # Estimate the point density
        z = estimate_point_density(dx, dy)

        ax = fig.add_subplot(2, 2, i + 1)
        # Plot the derivative of the center point (x, y)
        ax.scatter(dx, dy, c=z, s=100)
        ax.scatter(x_mean, y_mean, c="red", marker="x", s=50)

        size = 40
        ax.set_xlim(left=-size, right=size)
        ax.set_ylim(bottom=-size, top=size)
        ax.grid(True)

        ax.set_xlabel("dx")
        ax.set_ylabel("dy")
        ax.set_title(f"Center point derivatives for {person_id}")

    plt.show()
    fig.savefig(IDENTITY_DIR / "point_2d_derivatives.png")


def plot_2d_point_contour_derivatives():
    rows = []
    with open(IDENTITY_DIR / "identities.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    person_ids = ["person_id1", "person_id2", "person_id3", "person_id4"]

    fig = plt.figure(figsize=(10, 15), tight_layout=True)

    for i, person_id in enumerate(person_ids):

        # Filter the data to get only the rows for person_id1
        person_data = [row for row in rows if row["ClassID"] == person_id]

        # Compute the center point (x, y) of the bounding box for each frame
        x, y = [], []
        for row in person_data:
            x.append(compute_center_point(row["XMin"], row["XMax"]))
            y.append(compute_center_point(row["YMin"], row["YMax"]))

        # Compute the derivative of the center point (x, y) over different frames
        dx, dy = compute_gradients(x), compute_gradients(y)

        xx, yy, f, xmin, xmax, ymin, ymax = perpare_data(dx, dy)

        ax = fig.add_subplot(2, 2, i + 1)
        ax.imshow(np.rot90(f), extent=[xmin, xmax, ymin, ymax])
        cset = ax.contour(xx, yy, f, colors="k")
        ax.clabel(cset, inline=1, fontsize=5)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.title(f"2D Gaussian KDE for {person_id}")

    plt.show()
    fig.savefig(IDENTITY_DIR / "point_2d_contour_derivatives.png")


def plot_3d_point_derivatives():
    rows = []
    with open(IDENTITY_DIR / "identities.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    person_ids = ["person_id1", "person_id2", "person_id3", "person_id4"]

    fig = plt.figure(figsize=(10, 15), tight_layout=True)

    for i, person_id in enumerate(person_ids):

        # Filter the data to get only the rows for person_id1
        person_data = [row for row in rows if row["ClassID"] == person_id]

        # Compute the center point (x, y) of the bounding box for each frame
        x, y = [], []
        for row in person_data:
            x.append(compute_center_point(row["XMin"], row["XMax"]))
            y.append(compute_center_point(row["YMin"], row["YMax"]))

        # Compute the derivative of the center point (x, y) over different frames
        dx, dy = compute_gradients(x), compute_gradients(y)

        xx, yy, f, _, _, _, _ = perpare_data(dx, dy)

        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        surf = ax.plot_surface(
            xx, yy, f, rstride=1, cstride=1, cmap="coolwarm", edgecolor="none"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("PDF")
        ax.set_title(f"3D Gaussian KDE for {person_id}")
        fig.colorbar(surf, shrink=0.5, aspect=5)  # add color bar indicating the PDF
        ax.view_init(30, 35)

    plt.show()
    fig.savefig(IDENTITY_DIR / "point_3d_derivatives.png")


def plot_point_positions():
    # Read the CSV file into a list of dictionaries
    rows = []
    with open(IDENTITY_DIR / "identities.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    person_ids = ["person_id1", "person_id2", "person_id3", "person_id4"]

    fig = plt.figure(figsize=(10, 15), tight_layout=True)

    for i, person_id in enumerate(person_ids):

        # Filter the data to get only the rows for person_id1
        person_data = [row for row in rows if row["ClassID"] == person_id]

        # Compute the center point (x, y) of the bounding box for each frame
        # TODO: points are flipped!
        x, y = [], []
        for row in person_data:
            x.append(compute_center_point(row["XMin"], row["XMax"]))
            y.append(compute_center_point(row["YMin"], row["YMax"]))

        ax = fig.add_subplot(2, 2, i + 1)
        # Plot the derivative of the center point (x, y)
        ax.scatter(x, y, s=5)
        ax.set_xlim(left=0, right=1848)
        ax.set_ylim(bottom=0, top=1053)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Center point for {person_id} over different frames")

    plt.show()
    fig.savefig(IDENTITY_DIR / "point_motions.png")


if __name__ == "__main__":
    plot_point_derivatives()
    plot_2d_point_contour_derivatives()
    plot_2d_point_derivatives()
    plot_3d_point_derivatives()
    plot_point_positions()
