import os
import shutil
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

parent_folder = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
)
sys.path.append(parent_folder)

from src.emotion.features.extractors.face_embedder import (
    FaceEmbedder,
    create_face_embedder,
)
from src.emotion.utils.constants import DATA_DIR_TEST_IMAGES


def scree_plot(X: np.ndarray) -> int:
    """Plots the scree plot for the PCA."""
    # Fit PCA model
    pca = PCA()
    pca.fit(X)

    # Extract explained variances
    explained_variances = pca.explained_variance_ratio_

    # Calculate broken-stick model
    bs = 1.0 / np.arange(1, len(explained_variances) + 1)

    # Calculate cumulative variances
    cumulative_variances = np.cumsum(explained_variances)

    # Calculate residuals
    residuals = -(explained_variances - bs)

    # Find elbow point
    threshold = 0.2
    elbow_point = np.argmax(residuals < threshold)

    # Plot scree plot
    _, ax = plt.subplots()
    ax.plot(
        np.arange(1, len(explained_variances) + 1),
        explained_variances,
        "bo-",
        linewidth=2,
    )
    ax.plot(np.arange(1, len(explained_variances) + 1), bs, "r--", linewidth=2)
    ax.plot(
        np.arange(1, len(explained_variances) + 1),
        cumulative_variances,
        "g--",
        linewidth=2,
    )
    ax.set_title("Scree Plot")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Proportion of Variance Explained")
    ax.legend(["Actual", "Broken-Stick", "Cumulative"])

    # Plot elbow point
    ax.plot(
        elbow_point + 1,
        explained_variances[elbow_point],
        "ro",
        markersize=10,
        label=f"Elbow Point ({elbow_point+1})",
    )

    plt.show()

    return elbow_point


def cluster_data(X: np.ndarray, K: int, elbow: int) -> Tuple[np.ndarray, np.ndarray]:
    """Performs K-means clustering on the data using the given number of clusters."""
    # Reduce dimensions with PCA
    pca = PCA(n_components=elbow)
    reduced_dims = pca.fit_transform(X)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=K, random_state=0)
    labels = kmeans.fit_predict(reduced_dims)

    return labels, reduced_dims


if __name__ == "__main__":
    # images_path = [item for item in DATA_DIR_TEST_IMAGES.iterdir() if item.is_dir()]
    images_path = DATA_DIR_TEST_IMAGES
    K = 4

    # embedder = create_face_embedder({"type": "facerecog"})
    embedder = create_face_embedder(
        {"type": "insightface", "ctx_id": 0, "det_size": 128}
    )

    # Load the high-dimensional embeddings into a numpy array
    # X_embedded, image_names = generate_face_embeddings(images_path, embedder=embedder)
    embeddings = embedder.get_face_embeddings_from_folder_pca(images_path)

    X_embedded = np.array([embedding["embedding"] for embedding in embeddings])

    image_names = [embedding["image_path"] for embedding in embeddings]

    elbow = scree_plot(X_embedded) + 1

    use_elbow = input(f"Do you want to use the detected elbow point: {elbow}? (y/n)")

    if use_elbow == "n":
        new_elbow = int(input("Please enter the new elbow point: "))
        elbow = new_elbow

    labels, reduced_dims = cluster_data(X_embedded, K, elbow)

    # Define the colors for each cluster
    colors = plt.cm.get_cmap("viridis", K)

    # Create a scatter plot for each cluster
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for i in range(K):
        cluster_i = np.where(labels == i)[0]
        ax.scatter(
            reduced_dims[cluster_i, 0],
            reduced_dims[cluster_i, 1],
            reduced_dims[cluster_i, 2],
            c=colors(i),
            alpha=0.8,
            label=f"Cluster {i+1}",
        )

    # Add axis labels and legend
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    ax.legend()

    for i in range(K):
        cluster_i = np.where(labels == i)[0]
        cluster_dir = images_path / f"person_id{i+1}"
        cluster_dir.mkdir(exist_ok=True)
        for j in cluster_i:
            embedding = embeddings[j]
            image_path = embedding["image_path"]
            image_filename = image_path.name
            cluster_image_path = cluster_dir / image_filename
            shutil.copy(str(image_path), str(cluster_image_path))
            image_path.unlink()

    # Show the plot
    plt.show()
