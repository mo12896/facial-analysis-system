from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

DATA_DIR_IMAGES = Path("/home/moritz/Workspace/masterthesis/data/images")


def get_face_embeddings(images_path: Path) -> list:
    """Returns the mean embeddings of an identity.

    Args:
        images_path (Path): Path to the images of a person.

    Returns:
        np.ndarray: Mean embedding of the person.
    """
    model = FaceAnalysis()
    model.prepare(ctx_id=0, det_size=(128, 128))

    images = [cv2.imread(str(image)) for image in images_path.glob("*.png")]

    # Predict the faces
    faces = [model.get(img)[0] for img in images]
    # Fetch the embeddings
    embeddings = [face.normed_embedding for face in faces]

    return embeddings


def generate_face_embeddings(images_path: list[Path]) -> np.ndarray:
    """Generates the embeddings for a set of identities.

    Args:
        images_path (list[Path]): List of paths to the images of a person.

    Returns:
        list[np.ndarray]: List of embeddings.
    """

    final_embeddings = []
    for image_path in images_path:
        embeddings = get_face_embeddings(image_path)
        final_embeddings += embeddings

    return np.array(final_embeddings, dtype=np.float32)


if __name__ == "__main__":
    images_path = [item for item in DATA_DIR_IMAGES.iterdir() if item.is_dir()]
    # Best results with t-SNE 2 and PCA 2 or 3
    t_sne = 2
    n_pca = 3

    # Load the high-dimensional embeddings into a numpy array
    X_embedded = generate_face_embeddings(images_path)

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
