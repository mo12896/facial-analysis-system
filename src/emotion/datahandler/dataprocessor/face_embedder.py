import functools
from abc import ABC, abstractmethod
from pathlib import Path

import cv2

# Run f-r from seperate env, because dlib and cuda are no friends
# import face_recognition
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis

from src.emotion.utils.detections import Detections
from src.emotion.utils.utils import timer


class FaceEmbedder(ABC):
    """Abstract class for face embedding extraction."""

    @abstractmethod
    def __init__(self, parameters: dict = {}):
        self.parameters = parameters

    @abstractmethod
    def get_face_embeddings(
        self, detections: Detections, image: np.ndarray
    ) -> pd.DataFrame:
        """Retrieve the face embeddings for a set of detections.

        Args:
            detections (Detections): Set of detections stored in a Detections object.
            image (np.ndarray): The current frame of the video.

        Returns:
            pd.DataFrame: A Dataframe storing the embeddings for each detection.
        """

    @abstractmethod
    def get_face_embeddings_from_folder(self, image_folder: Path) -> list:
        """Returns the embeddings of an identity.

        Args:
            images_path (Path): Path to the images of a person.

        Returns:
            np.ndarray: Embeddings of the person.
        """

    @timer
    def get_anchor_face_embedding(self, image_folder: Path) -> np.ndarray:
        """Returns the mean embedding of an identity.

        Args:
            images_path (Path): Path to the images of a person.

        Returns:
            np.ndarray: Mean embedding of the person.
        """

        @functools.wraps(self.get_face_embeddings_from_folder)
        def wrapper(image_folder: Path) -> np.ndarray:
            result = self.get_face_embeddings_from_folder(image_folder)

            embedding = np.mean(np.array(result, dtype=np.float32), axis=0)

            return embedding

        return wrapper(image_folder)


def create_face_embedder(parameters: dict) -> FaceEmbedder:
    """Create a face embedder based on the given parameters.

    Args:
        parameters (dict): A dictionary containing the parameters for the face embedder.

    Returns:
        FaceEmbedder: A face embedder object.
    """

    if parameters["type"] == "insightface":
        return InsightFaceEmbedder(parameters)
    elif parameters["type"] == "facerecog":
        raise ValueError("Currently not supported.")
    else:
        raise ValueError("Unknown face embedder type.")


# TODO: Try different models from their model zoo!
class InsightFaceEmbedder(FaceEmbedder):
    """Wrapper around the insightface model for face embedding extraction."""

    def __init__(self, parameters: dict = {}):
        super().__init__(parameters)
        self.model = FaceAnalysis()

        ctx_id = parameters.get("ctx_id", 0)
        det_size = parameters.get("det_size", 128)
        self.model.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))

    @timer
    def get_face_embeddings(
        self, detections: Detections, image: np.ndarray
    ) -> pd.DataFrame:
        """Get the face embeddings for a set of detections using the insightface model."""

        data = {}

        for key, bbox in zip(detections.class_id, detections.bboxes):
            img = image[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
            face = self.model.get(img)
            if len(face) == 0:
                continue
            embedding = np.array(face[0].normed_embedding, dtype=np.float32)
            data[key] = embedding

        return pd.DataFrame(data).transpose()

    @timer
    def get_face_embeddings_from_folder(self, image_folder: Path) -> list:
        embeddings = []

        for image in image_folder.glob("*.png"):
            img = cv2.imread(str(image))
            face = self.model.get(img)
            if len(face) == 0:
                continue
            embeddings.append(face[0].normed_embedding)

        return embeddings


# Only 128D embeddings, thus worse differentiation between identities.
# Thus, not relevant for now!
# class FaceRecognitionEmbedder(FaceEmbedder):
#     def __init__(self, parameters: dict = {}):
#         super().__init__(parameters)

#     @timer
#     def get_face_embeddings(
#         self, detections: Detections, image: np.ndarray
#     ) -> pd.DataFrame:

#         data = {}

#         for key, bbox in zip(detections.class_id, detections.bboxes):
#             image = image[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
#             height, width, _ = image.shape
#             bbox = [(0, height, width, 0)]
#             embedding = face_recognition.face_encodings(
#                 face_image=image, num_jitters=1, known_face_locations=bbox
#             )
#             if len(embedding) == 0:
#                 continue
#             data[key] = embedding[0]

#         return pd.DataFrame(data).transpose()

#     @timer
#     def get_face_embeddings_from_folder(self, images_path: Path) -> list:
#         embeddings = []

#         for image in images_path.glob("*.png"):
#             img = cv2.imread(str(image))
#             height, width, _ = img.shape
#             bbox = [(0, height, width, 0)]
#             embedding = face_recognition.face_encodings(
#                 face_image=img, num_jitters=1, known_face_locations=bbox
#             )
#             if len(embedding) == 0:
#                 continue
#             embeddings.append(embedding[0])

#         return embeddings

if __name__ == "__main__":
    print("This is a module.")
