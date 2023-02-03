# import os
# import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.emotion.datahandler.dataprocessor.face_embedder import (
    FaceEmbedder,
    create_face_embedder,
)
from src.emotion.utils.constants import DATA_DIR_IMAGES

# parent_folder = os.path.abspath(
#     os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
# )
# sys.path.append(parent_folder)


def generate_face_embeddings(
    image_folders: list[Path], embedder: FaceEmbedder
) -> np.ndarray:
    """Generates the embeddings for a set of identities.

    Args:
        images_path (list[Path]): List of paths to the images of a person.

    Returns:
        list[np.ndarray]: List of embeddings.
    """

    final_embeddings = []
    for image_folder in image_folders:
        embeddings = embedder.get_face_embeddings_from_folder(image_folder)
        final_embeddings += embeddings

    return np.array(final_embeddings, dtype=np.float32)


if __name__ == "__main__":
    images_path = [item for item in DATA_DIR_IMAGES.iterdir() if item.is_dir()]
    # Best results with t-SNE 2 and PCA 2 or 3
    t_sne = 2
    n_pca = 3

    embedder = create_face_embedder(
        {"type": "insightface", "ctx_id": 0, "det_size": 128}
    )

    # Load the high-dimensional embeddings into a numpy array
    X_embedded = generate_face_embeddings(images_path, embedder=embedder)

    if t_sne == 2:
        tsne = TSNE(n_components=2)
        embeddings_2d = tsne.fit_transform(X_embedded)

        # Plot the results
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
        plt.show()
    elif t_sne == 3:
        tsne = TSNE(n_components=3)
        embeddings_3d = tsne.fit_transform(X_embedded)

        # Plot the 3D embeddings
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2])
        plt.show()
    else:
        raise ValueError("t-SNE must be 2 or 3.")

    if n_pca == 2:
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(X_embedded)

        # Plot the results
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
        plt.show()
    elif n_pca == 3:
        pca = PCA(n_components=3)
        embeddings_3d = pca.fit_transform(X_embedded)

        # Plot the 3D embeddings
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2])
        plt.show()
    else:
        raise ValueError("PCA must be 2 or 3.")
