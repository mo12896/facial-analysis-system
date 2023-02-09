# import os
# import sys
from abc import ABC, abstractmethod

import cv2
import matplotlib.pyplot as plt
import numpy as np

# grandparent_folder = os.path.abspath(
#     os.path.join(
#         os.path.dirname(os.path.abspath(__file__)),
#         os.pardir,
#         os.pardir,
#         os.pardir,
#         os.pardir,
#     )
# )
# sys.path.append(grandparent_folder)

from pytop.pyt_openpose.body import Body
from pytop.pyt_openpose.hand import Hand
from src.emotion.datahandler.dataprocessor.face_detector import create_face_detector
from src.emotion.utils.color import Color
from src.emotion.utils.constants import MODEL_DIR
from src.emotion.utils.detections import Detections
from src.emotion.utils.keypoint_annotator import KeyPointAnnotator
from src.emotion.utils.utils import timer


class PoseEstimator(ABC):
    """Abstract class for pose estimation."""

    @abstractmethod
    def __init__(self, parameters: dict = {}) -> None:
        """Constructer for the PoseEstimator class.

        Args:
            parameters (dict, optional): Dictionary containing the parameters for the pose.
        """
        self.parameters = parameters

    @abstractmethod
    def estimate_poses(self, image: np.ndarray, detections: Detections) -> Detections:
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


class PyTorchOpenPoseEstimator(PoseEstimator):
    def __init__(self, parameters: dict = {}) -> None:
        super().__init__(parameters)
        # Load OpenPose models
        body_pose = MODEL_DIR / "body_pose_model.pth"
        hand_pose = MODEL_DIR / "hand_pose_model.pth"
        self.body_pose_estimator = Body(body_pose)
        self.hand_pose_estimator = Hand(hand_pose)
        self.hands = True
        self.body = True

    # Unfortunately, pretty slow (~1.5sec per frame) -> 5x video length
    @timer
    def estimate_poses(self, image: np.ndarray, detections: Detections) -> Detections:
        if self.body:
            candidate, subset = self.body_pose_estimator(image)
        # TODO; Write parser for this!
        # if self.hands:
        #     hands_list = util.handDetect(candidate, subset, image)
        #     all_hand_peaks = []
        #     for x, y, w, is_left in hands_list:
        #         peaks = self.hand_pose_estimator(image[y : y + w, x : x + w, :])
        #         peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
        #         peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
        #         all_hand_peaks.append(peaks)

        detections = detections.poses_from_pytorch_openpose(candidate, subset)

        return detections


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


class OpenPoseEstimator(PoseEstimator):
    def __init__(self, parameters: dict = {}) -> None:
        super().__init__(parameters)
        # Load OpenPose model
        proto_file = "path/to/pose/coco/pose_deploy_linevec.prototxt"
        weights_file = "path/to/pose/coco/pose_iter_440000.caffemodel"
        self.pose_estimator = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

    @timer
    def estimate_poses(self, image: np.ndarray):

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

        return all_keypoints

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


if __name__ == "__main__":

    # Create a pose estimator
    pose_estimator = create_pose_estimator({"type": "pytorch"})
    image = cv2.imread("/home/moritz/Workspace/masterthesis/data/test_image.png")

    face_detector = create_face_detector("retinaface")
    detections = face_detector.detect_faces(image)

    detections = pose_estimator.estimate_poses(image, detections)

    keypoints_annotator = KeyPointAnnotator(color=Color.red())
    image = keypoints_annotator.annotate_pytorch_openpose(image, detections)

    # canvas = util.draw_bodypose(image, result[0], result[1])
    # plt.imshow(image[:, :, [2, 1, 0]])
    plt.imshow(image)
    plt.show()
