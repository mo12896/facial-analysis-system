from abc import ABC, abstractmethod

import cv2
import numpy as np
from src.body import Body
from src.hand import Hand

from src import util
from src.emotion.utils.constants import MODEL_DIR
from src.emotion.utils.detections import Detections


class PoseEstimator(ABC):
    @abstractmethod
    def __init__(self, parameters: dict = {}) -> None:
        """Constructer for the PoseEstimator class.

        Args:
            parameters (dict, optional): Dictionary containing the parameters for the pose.
        """
        self.parameters = parameters

    @abstractmethod
    def estimate_pose(self, detections: Detections, image: np.ndarray) -> Detections:
        """Detect keypoints for a set of detections.

        Args:
            image (np.ndarray): The current frame of the video.

        Returns:
            Detections: Detection object with the keypoints.
        """


def create_pose_estimator(parameters: dict = {}) -> PoseEstimator:
    """Factory method for creating a pose estimator.

    Args:
        parameters (dict, optional): Dictionary containing the parameters for the
        pose estimator. Defaults to {}.

    Raises:
        ValueError: Raised if the pose estimator is not supported.

    Returns:
        PoseEstimator: Pose estimator object.
    """
    if parameters["type"] == "openpose":
        return OpenPoseEstimator(parameters)
    elif parameters["type"] == "pytorch":
        return PyTorchOpenPoseEstimator(parameters)
    else:
        raise ValueError(f"Pose estimator {parameters['type']} is not supported")


def PyTorchOpenPoseEstimator(PoseEstimator):
    def __init__(self, parameters: dict = {}) -> None:
        super().__init__(parameters)
        # Load OpenPose models
        body_pose = MODEL_DIR / "body_pose_model.pth"
        hand_pose = MODEL_DIR / "hand_pose_model.pth"
        self.body_pose_estimator = Body(body_pose)
        self.hand_pose_estimator = Hand(hand_pose)
        self.hands = True
        self.body = True

    def estimate_pose(self, detections: Detections, image: np.ndarray) -> Detections:
        if self.body:
            candidate, subset = self.body_estimation(image)
        if self.hands:
            hands_list = util.handDetect(candidate, subset, image)
            all_hand_peaks = []
            for x, y, w, is_left in hands_list:
                peaks = self.hand_estimation(image[y : y + w, x : x + w, :])
                peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
                peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
                all_hand_peaks.append(peaks)

            # TODO: Write parser for this!
            # detections = detections.match_keypoints(all_keypoints)

        return detections


def OpenPoseEstimator(PoseEstimator):
    def __init__(self, parameters: dict = {}) -> None:
        super().__init__(parameters)
        # Load OpenPose model
        proto_file = "path/to/pose/coco/pose_deploy_linevec.prototxt"
        weights_file = "path/to/pose/coco/pose_iter_440000.caffemodel"
        self.pose_estimator = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

    def estimate_pose(self, detections: Detections, image: np.ndarray) -> Detections:

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

        # TODO: Write the parser!
        # detections = detections.match_keypoints(all_keypoints)

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
