from abc import ABC, abstractmethod

import cv2
import numpy as np

from src.emotion.utils.detections import Detections


class PoseEstimator(ABC):
    def __init__(self, parameters: dict = {}) -> None:
        """Constructor for the PoseEstimator class."""
        self.parameters = parameters

    @abstractmethod
    def detect_keypoints(self, image: np.ndarray) -> Detections:
        """Detect keypoints for a set of detections.

        Args:
            image (np.ndarray): The current frame of the video.

        Returns:
            Detections: Detection object with the keypoints.
        """


def OpenPoseEstimator(PoseEstimator):
    def __init__(self, parameters: dict = {}) -> None:
        super().__init__(parameters)
        # Load OpenPose model
        proto_file = "path/to/pose/coco/pose_deploy_linevec.prototxt"
        weights_file = "path/to/pose/coco/pose_iter_440000.caffemodel"
        self.pose_estimator = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

    def detect_keypoints(self, image: np.ndarray, detections: Detections) -> Detections:

        # Create a 4D blob from the image
        blob = cv2.dnn.blobFromImage(
            image, 1.0 / 255, (640, 480), (0, 0, 0), swapRB=False, crop=False
        )

        self.pose_estimator.setInput(blob)

        output = self.pose_estimator.forward()

        # Shape of output is 1x80x49x49, where 80 is the number of keypoints
        # The first 49x49 is a heatmap, and the second 49x49 is part affinity fields
        heatmap = output[:, :18, :, :]

        heatmap = cv2.resize(heatmap[0], (image.shape[1], image.shape[0]))

        # Initialize a list to store keypoints for all persons
        all_keypoints = []

        for i in range(heatmap.shape[2]):
            person_heatmap = heatmap[:, :, i]
            person_keypoints = self.extract_keypoints(person_heatmap, image)
            all_keypoints.append(person_keypoints)

        # TODO: Implement
        detections = detections.match_keypoints(all_keypoints)

        return detections

    @staticmethod
    def extract_keypoints(heatmap: np.ndarray, img: np.ndarray) -> list[tuple[int]]:
        """Function to extract keypoints for a single person

        Args:
            heatmap (np.ndarray): Heatmap for a single person
            img (np.ndarray): The current frame of the video

        Returns:
            list[tuple[int]]: List of keypoints for a single person
        """
        coordinates = []
        for i in range(18):
            _, conf, _, point = cv2.minMaxLoc(heatmap[:, :, i])
            x = (img.shape[1] * point[0]) / heatmap.shape[1]
            y = (img.shape[0] * point[1]) / heatmap.shape[0]
            coordinates.append((x, y))

        return coordinates


# from emotion.models.body.light_openpose.load_state import load_state
# from emotion.models.body.light_openpose.with_mobilenet import (
#     PoseEstimationWithMobileNet,
# )
# from emotion.utils.constants import LIGHT_OPENPOSE_MODEL


# class PoseEstimator(ABC):
#     def __init__(self, model_path: str, cpu: str) -> None:
#         self.model_path = model_path
#         self.cpu = cpu

#     @abstractmethod
#     def construct_model(self) -> None:
#         pass


# def create_pose_estimator(estimator: str) -> PoseEstimator:
#     """Factory method to create pose estimator objects."""
#     if estimator == "light_openpose":
#         return LightOpenPoseEstimator()
#     else:
#         raise NotImplementedError


# class LightOpenPoseEstimator(PoseEstimator):
#     def __init__(self, model_path=LIGHT_OPENPOSE_MODEL, cpu=False) -> None:
#         super().__init__(model_path, cpu)

#     def construct_model(self) -> PoseEstimationWithMobileNet:
#         self.net = PoseEstimationWithMobileNet()
#         checkpoint = torch.load(self.model_path, map_location="cpu")
#         load_state(self.net, checkpoint)
#         self.net.eval()
#         if not self.cpu:
#             self.net = self.net.cuda()
#         return self.net
