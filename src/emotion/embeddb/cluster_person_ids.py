# import os
import shutil

# import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# from sklearn_extra.cluster import KMedoids
from tensorflow.keras import layers

from src.emotion.features.extractors.face_embedder import create_face_embedder
from src.emotion.utils.constants import DATA_DIR_TEST_IMAGES

# parent_folder = os.path.abspath(
#     os.path.join(
#         os.path.dirname(os.path.abspath(__file__)),
#         os.pardir,
#         os.pardir,
#         os.pardir,
#     )
# )
# sys.path.append(parent_folder)


class DimensionalityReducer(ABC):
    """Abstract base class for dimensionality reduction methods."""

    @abstractmethod
    def reduce_dims(self, X: np.ndarray) -> np.ndarray:
        """Reduces the dimensions of the input data X."""
        pass


class PCAReducer(DimensionalityReducer):
    """Dimensionality reduction using PCA."""

    def __init__(self, n_components: int):
        self.pca = PCA(n_components=n_components)

    def reduce_dims(self, X: np.ndarray) -> np.ndarray:
        return self.pca.fit_transform(X)


class AutoencoderReducer(DimensionalityReducer):
    """Dimensionality reduction using autoencoder."""

    def __init__(self, encoding_dim: int, epochs: int = 50, batch_size: int = 32):
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.autoencoder = None
        self.encoder = None

    def reduce_dims(self, X: np.ndarray) -> np.ndarray:
        if self.autoencoder is None:
            self.autoencoder = self.create_autoencoder(X.shape[1], self.encoding_dim)
            self.autoencoder.fit(
                X, X, epochs=self.epochs, batch_size=self.batch_size, verbose=0
            )
            self.encoder = tf.keras.Model(
                self.autoencoder.input, self.autoencoder.layers[-2].output
            )
        return self.encoder(X).numpy()

    @staticmethod
    def create_autoencoder(input_dim: int, encoding_dim: int) -> tf.keras.Model:
        """Creates an autoencoder model with the given input and encoding dimensions."""
        input_layer = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(encoding_dim, activation="relu")(input_layer)
        decoded = layers.Dense(input_dim, activation="sigmoid")(encoded)
        autoencoder = tf.keras.Model(input_layer, decoded)
        autoencoder.compile(optimizer="adam", loss="mse")
        return autoencoder


def create_dimensionality_reducer(params: dict = {}) -> DimensionalityReducer:
    """Creates a dimensionality reducer based on the given method."""
    method = params.get("type", "pca")
    n_components = params.get("n_components", 4)
    encoding_dim = params.get("encoding_dim", 32)
    if method == "pca":
        return PCAReducer(n_components)
    elif method == "autoencoder":
        return AutoencoderReducer(encoding_dim)
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")


class Clusterer(ABC):
    """Abstract base class for clustering algorithms."""

    @abstractmethod
    def cluster(self, X: np.ndarray) -> np.ndarray:
        """Performs clustering on the input data X."""
        pass


class KMeansClusterer(Clusterer):
    """Clustering using K-means."""

    def __init__(self, K: int):
        self.K = K
        self.kmeans = None

    def cluster(self, X: np.ndarray) -> np.ndarray:
        if self.kmeans is None:
            self.kmeans = KMeans(n_clusters=self.K, random_state=0)
        return self.kmeans.fit_predict(X)


# class KMedoidsClusterer(Clusterer):
#     """Clustering using K-medoids."""

#     def __init__(self, K: int):
#         self.K = K
#         self.kmedoids = None

#     def cluster(self, X: np.ndarray) -> np.ndarray:
#         if self.kmedoids is None:
#             self.kmedoids = KMedoids(n_clusters=self.K, random_state=0)
#         return self.kmedoids.fit_predict(X)


def create_clusterer(params: dict = {}) -> Clusterer:
    """Creates a clustering algorithm based on the given method."""
    method = params.get("type", "kmeans")
    K = params.get("K", 4)
    if method == "kmeans":
        return KMeansClusterer(K)
    elif method == "kmedoids":
        raise NotImplementedError("K-medoids clustering is not implemented yet.")
    else:
        raise ValueError(f"Unknown clustering method: {method}")


class ClustererAdapter:
    """Adapter class that wires up the dimensionality reducer and the clustering algorithm."""

    def __init__(self, reducer: DimensionalityReducer, clusterer: Clusterer):
        self.reducer = reducer
        self.clusterer = clusterer

    def cluster_data(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Performs clustering on the input data X using the dimensionality reducer and the clustering algorithm."""
        reduced_X = self.reducer.reduce_dims(X)
        labels = self.clusterer.cluster(reduced_X)
        return labels, reduced_X


def compute_pca_metrics(
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Computes relevant metrics for PCA scree plot"""
    pca = PCA()
    pca.fit(X)

    explained_variances = pca.explained_variance_ratio_
    bs = 1.0 / np.arange(1, len(explained_variances) + 1)
    cumulative_variances = np.cumsum(explained_variances)
    residuals = -(explained_variances - bs)
    threshold = 0.2
    elbow_point = np.argmax(residuals < threshold) + 1

    return explained_variances, bs, cumulative_variances, elbow_point


def scree_plot(
    explained_variances: np.ndarray,
    bs: np.ndarray,
    cumulative_variances: np.ndarray,
    elbow_point: int,
):
    """Plots the scree plot for the PCA."""
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
        elbow_point,
        explained_variances[elbow_point],
        marker="*",
        color="red",
        markersize=15,
        label=f"Elbow Point ({elbow_point})",
    )

    # Add text annotation for elbow count
    ax.text(
        elbow_point,
        explained_variances[elbow_point],
        f" Elbow Point ({elbow_point})",
        fontsize=12,
    )

    plt.show()


def plot_clusters(labels: np.ndarray, reduced_dims: np.ndarray, K: int):
    """
    Create a 3D scatter plot for each cluster.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.get_cmap("viridis", K)
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

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    ax.legend()

    plt.show()


def save_clusters(
    labels: np.ndarray, embeddings: list, images_path: Path, K: int
) -> None:
    """
    Save the clustered images to their respective directories.
    """
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


if __name__ == "__main__":
    # images_path = [item for item in DATA_DIR_TEST_IMAGES.iterdir() if item.is_dir()]
    images_path = DATA_DIR_TEST_IMAGES
    K = 4
    SAVE_CLUSTERS = False

    # embedder = create_face_embedder({"type": "facerecog"})
    embedder = create_face_embedder(
        {"type": "insightface", "ctx_id": 0, "det_size": 128}
    )

    # Load the high-dimensional embeddings into a numpy array
    embeddings = embedder.get_face_embeddings_from_folder(images_path)
    X_embedded = np.array([embedding["embedding"] for embedding in embeddings])

    # Compute and plot the PCA metrics
    explained_variances, bs, cumulative_variances, elbow = compute_pca_metrics(
        X_embedded
    )
    scree_plot(explained_variances, bs, cumulative_variances, elbow)

    use_elbow = input(f"Do you want to use the detected elbow point: {elbow}? (y/n)")
    if use_elbow == "n":
        new_elbow = int(input("Please enter the new elbow point: "))
        elbow = new_elbow

    # Create the clusters
    reducer = create_dimensionality_reducer({"type": "pca", "encoding_dim": elbow})
    clusterer = create_clusterer({"type": "kmeans", "K": K})
    adapter = ClustererAdapter(reducer, clusterer)
    labels, reduced_dims = adapter.cluster_data(X_embedded)

    plot_clusters(labels, reduced_dims, K)

    # Save the clusters
    if SAVE_CLUSTERS:
        save_clusters(labels, embeddings, images_path, K)
