from typing import Any

import numpy as np
import yaml

from src.emotion.embeddb.cluster_person_ids import (
    ClustererAdapter,
    compute_pca_metrics,
    create_clusterer,
    create_dimensionality_reducer,
    plot_clusters,
    save_clusters,
)
from src.emotion.embeddb.crop_random_faces import (
    cleanup,
    crop_random_faces_from_n_frames,
)
from src.emotion.embeddb.save_embeddings import (
    generate_face_embeddings,
    validate_embeddings,
    write_embeddings_to_database,
)
from src.emotion.features.extractors.face_detector import create_face_detector
from src.emotion.features.extractors.face_embedder import create_face_embedder
from src.emotion.utils.constants import (
    CONFIG_DIR,
    DATA_DIR,
    DATA_DIR_DATABASE,
    DATA_DIR_TEST_IMAGES,
)
from src.emotion.utils.utils import SQLite

try:
    configs_dir = CONFIG_DIR / "config.yaml"
    configs: dict[str, Any] = yaml.safe_load(configs_dir.read_text())
    print("Loaded config file into python dict!")
except yaml.YAMLError as exc:
    print(exc)

detector_params = configs.get("DETECTOR", "scrfd")
embedder_params = configs.get("EMBEDDER", "insightface")
sample_frames = configs.get("SAMPLE_FRAMES", 20)
database = DATA_DIR_DATABASE / configs.get("DATABASE", "embeddings_test.db")
reducer_params = configs.get("REDUCER", {"type": "pca", "n_components": 4})
clusterer_params = configs.get("CLUSTERER", {"type": "kmeans", "K": 4})
save_embeddings = configs.get("SAVE_EMBEDDINGS", True)
K = configs.get("K", 4)
video_path = str(DATA_DIR / configs.get("VIDEO", "short_clip_debug.mp4"))
output_folder = DATA_DIR_TEST_IMAGES

if __name__ == "__main__":

    if output_folder.exists():
        response = input(f"{output_folder} already exists. Overwrite? [y/n] ")
        if response != "y":
            print("Script stopped by user.")
            exit()
        cleanup(output_folder)

    detector = create_face_detector(detector_params)

    crop_random_faces_from_n_frames(
        video_path, output_folder, detector, num_frames=sample_frames
    )

    embedder = create_face_embedder(embedder_params)

    # Load the high-dimensional embeddings into a numpy array
    embeddings = embedder.get_face_embeddings_from_folder(output_folder)
    X_embedded = np.array([embedding["embedding"] for embedding in embeddings])

    # Compute and plot the PCA metrics
    _, _, _, elbow = compute_pca_metrics(X_embedded)

    use_elbow = input(f"Do you want to use the detected elbow point: {elbow}? [y/n] ")
    if use_elbow == "n":
        new_elbow = int(input("Please enter the new elbow point: "))
        elbow = new_elbow

    reducer_params["n_components"] = elbow
    # Create the clusters
    reducer = create_dimensionality_reducer(reducer_params)
    clusterer = create_clusterer(clusterer_params)
    adapter = ClustererAdapter(reducer, clusterer)
    labels, reduced_dims = adapter.cluster_data(X_embedded)

    plot_clusters(labels, reduced_dims, K)

    # Save the clusters
    if save_embeddings:
        save_clusters(labels, embeddings, output_folder, K)

        images_path = [item for item in DATA_DIR_TEST_IMAGES.iterdir() if item.is_dir()]

        # Save the embeddings to the database
        if database.exists():
            response = input(f"{database} already exists. Overwrite? [y/n] ")
            if response != "y":
                exit()
            database.unlink()

        embeddings = generate_face_embeddings(
            images_path=images_path, embedder=embedder
        )
        write_embeddings_to_database(database=database, embeddings=embeddings)

        validate_embeddings(
            database=database, images_path=images_path, embedder=embedder
        )

        with SQLite(str(database)) as conn:
            conn.execute("SELECT person_id, embedding FROM embeddings")
            data = {
                person_id: np.frombuffer(embedding, dtype=np.float32)
                for person_id, embedding in conn
            }
            print(data)
