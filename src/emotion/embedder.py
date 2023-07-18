# import argparse
from typing import Any, Dict

import numpy as np
import yaml

from src.emotion.embeddb.cluster_person_ids import (
    ClustererAdapter,
    compute_pca_metrics,
    create_clusterer,
    create_dimensionality_reducer,
    plot_clusters,
    save_clusters,
    scree_plot,
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
from src.emotion.utils.constants import CONFIG_DIR, DATA_DIR_INPUT, DATA_DIR_OUTPUT
from src.emotion.utils.utils import SQLite


def load_configs(config_file: str = "config.yaml") -> Dict[str, Any]:
    try:
        configs_dir = CONFIG_DIR / config_file
        configs: Dict[str, Any] = yaml.safe_load(configs_dir.read_text())
        print("Loaded config file into python dict!")
        return configs
    except yaml.YAMLError as exc:
        print(exc)


class FaceClusterer:
    def __init__(self, params: Dict[str, Any]) -> None:
        video = params.get("VIDEO")

        if video is not None:
            self.database = DATA_DIR_OUTPUT / (
                str(video).split(".")[0]
                + "/utils/identity_templates_database/"
                + (str(video).split(".")[0] + ".db")
            )
            self.output_folder = DATA_DIR_OUTPUT / (
                str(video).split(".")[0] + "/utils/identity_images"
            )
            self.video_path = str(DATA_DIR_INPUT / video)

        self.detector_params = params.get("DETECTOR", "scrfd")
        self.embedder_params = params.get("EMBEDDER", "insightface")
        self.sample_frames = params.get("SAMPLE_FRAMES", 20)

        self.reducer_params = params.get("REDUCER", {"type": "pca", "n_components": 4})
        self.clusterer_params = params.get("CLUSTERER", {"type": "kmeans", "K": 4})
        self.save_embeddings = params.get("SAVE_EMBEDDINGS", True)
        self.K = params.get("K", 4)

    def create_database(self, verbose: bool = False) -> None:
        if self.output_folder.exists():
            response = input(f"{self.output_folder} already exists. Overwrite? [y/n] ")
            if response != "y":
                print("Script stopped by user.")
                exit()
            cleanup(self.output_folder)
        else:
            self.output_folder.mkdir(parents=True, exist_ok=True)

        detector = create_face_detector(self.detector_params)

        crop_random_faces_from_n_frames(
            self.video_path, self.output_folder, detector, num_frames=self.sample_frames
        )

        embedder = create_face_embedder(self.embedder_params)

        # Load the high-dimensional embeddings into a numpy array
        embeddings = embedder.get_face_embeddings_from_folder(self.output_folder)
        X_embedded = np.array([embedding["embedding"] for embedding in embeddings])

        # Compute and plot the PCA metrics
        explained_variances, bs, cumulative_variances, elbow = compute_pca_metrics(
            X_embedded
        )

        if verbose:
            scree_plot(explained_variances, bs, cumulative_variances, elbow)

        use_elbow = input(
            f"Do you want to use the detected elbow point: {elbow}? [y/n] "
        )
        if use_elbow == "n":
            new_elbow = int(input("Please enter the new elbow point: "))
            elbow = new_elbow

        self.reducer_params["n_components"] = elbow
        # Create the clusters
        reducer = create_dimensionality_reducer(self.reducer_params)
        clusterer = create_clusterer(self.clusterer_params)
        adapter = ClustererAdapter(reducer, clusterer)
        labels, reduced_dims = adapter.cluster_data(X_embedded)

        if verbose:
            plot_clusters(labels, reduced_dims, self.K)

        # Save the clusters
        if self.save_embeddings:
            save_clusters(labels, embeddings, self.output_folder, self.K)

            # Note that we have to track all persons, to get the gaze feature. Later we will then discard the
            # persons that do not want to be tracked.
            input(
                f"Have you validated all generated IDs under {self.output_folder}? Press Enter to confirm: "
            )
            input("Are you really sure? Press Enter to confirm: ")

            images_paths = [
                item for item in self.output_folder.iterdir() if item.is_dir()
            ]

            # Save the embeddings to the database
            if self.database.exists():
                response = input(f"{self.database} already exists. Overwrite? [y/n] ")
                if response != "y":
                    exit()
                self.database.unlink()
            else:
                self.database.parent.mkdir(parents=True, exist_ok=True)

            embeddings = generate_face_embeddings(
                images_path=images_paths, embedder=embedder
            )
            write_embeddings_to_database(database=self.database, embeddings=embeddings)

            validate_embeddings(
                database=self.database, images_path=images_paths, embedder=embedder
            )

            with SQLite(str(self.database)) as conn:
                conn.execute("SELECT person_id, embedding FROM embeddings")
                data = {
                    person_id: np.frombuffer(embedding, dtype=np.float32)
                    for person_id, embedding in conn
                }
                print(data)


# if __name__ == "__main__":
#     # Initialize the parser
#     parser = argparse.ArgumentParser(description="A script for face clustering")

#     # Add the arguments
#     parser.add_argument("config_file", type=str, help="Name of the configuration file")
#     parser.add_argument(
#         "-v", "--verbose", action="store_true", help="Enable verbose output"
#     )

#     # Parse the arguments
#     args = parser.parse_args()

#     configs = load_configs(args.config_file)
#     face_clusterer = FaceClusterer(configs)
#     face_clusterer.create_database(verbose=args.verbose)
