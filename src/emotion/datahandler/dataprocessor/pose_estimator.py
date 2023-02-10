# import os
# import sys
from abc import ABC, abstractmethod

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from external.light_openpose.models.with_mobilenet import PoseEstimationWithMobileNet
from external.light_openpose.modules.keypoints import extract_keypoints, group_keypoints
from external.light_openpose.modules.load_state import load_state
from external.light_openpose.modules.pose import Pose
from external.light_openpose.val import normalize, pad_width

# TODO: Change back from symlink to external import!
from pytop.pyt_openpose.body import Body
from pytop.pyt_openpose.hand import Hand
from src.emotion.datahandler.dataprocessor.face_detector import create_face_detector
from src.emotion.utils.color import Color
from src.emotion.utils.constants import DATA_DIR, MODEL_DIR
from src.emotion.utils.detections import Detections
from src.emotion.utils.keypoint_annotator import KeyPointAnnotator
from src.emotion.utils.utils import timer

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
    if parameters["type"] == "l_openpose":
        return LightOpenPoseEstimator(parameters)
    elif parameters["type"] == "py_openpose":
        return PyTorchOpenPoseEstimator(parameters)
    elif parameters["type"] == "openpose":
        return OpenPoseEstimator(parameters)
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

        group_keypoints = []

        for n in range(len(subset)):
            person_keypoints = np.zeros((18, 2), dtype=np.float32)
            for i in range(18):
                index = int(subset[n][i])
                if index == -1:
                    person_keypoints[i] = np.array([-1, -1])
                    continue
                x, y = candidate[index][0:2]
                person_keypoints[i] = np.array([x, y])
            group_keypoints.append(person_keypoints)

        detections = detections.match_poses(group_keypoints)

        return detections


class LightOpenPoseEstimator(PoseEstimator):
    def __init__(self, parameters: dict = {}) -> None:
        super().__init__(parameters)
        self.model_path = MODEL_DIR / "checkpoint_iter_370000.pth"
        self.cpu = False
        self.net = PoseEstimationWithMobileNet()

        checkpoint = torch.load(self.model_path, map_location="cpu")
        load_state(self.net, checkpoint)

        self.net.eval()
        if not self.cpu:
            self.net = self.net.cuda()

        self.stride = 8
        self.upsample_ratio = 4
        self.num_keypoints = Pose.num_kpts
        self.height_size = 256

    # TODO: Debug and write keypoint parser
    @timer
    def estimate_poses(self, image: np.ndarray, detections: Detections) -> Detections:
        heatmaps, pafs, scale, pad = self.infer_fast(
            self.net,
            image,
            self.height_size,
            self.stride,
            self.upsample_ratio,
            self.cpu,
        )

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(self.num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(
                heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num
            )

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (
                all_keypoints[kpt_id, 0] * self.stride / self.upsample_ratio - pad[1]
            ) / scale
            all_keypoints[kpt_id, 1] = (
                all_keypoints[kpt_id, 1] * self.stride / self.upsample_ratio - pad[0]
            ) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((self.num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(self.num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(
                        all_keypoints[int(pose_entries[n][kpt_id]), 0]
                    )
                    pose_keypoints[kpt_id, 1] = int(
                        all_keypoints[int(pose_entries[n][kpt_id]), 1]
                    )
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

        final_keypoints = [pose.keypoints for pose in current_poses]

        detections = detections.match_poses(final_keypoints)

        return detections

    @staticmethod
    def infer_fast(
        net,
        img,
        net_input_height_size,
        stride,
        upsample_ratio,
        cpu,
        pad_value=(0, 0, 0),
        img_mean=np.array([128, 128, 128], np.float32),
        img_scale=np.float32(1 / 256),
    ):
        height, width, _ = img.shape
        scale = net_input_height_size / height

        scaled_img = cv2.resize(
            img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
        )
        scaled_img = normalize(scaled_img, img_mean, img_scale)
        min_dims = [
            net_input_height_size,
            max(scaled_img.shape[1], net_input_height_size),
        ]
        padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
        if not cpu:
            tensor_img = tensor_img.cuda()

        stages_output = net(tensor_img)

        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(
            heatmaps,
            (0, 0),
            fx=upsample_ratio,
            fy=upsample_ratio,
            interpolation=cv2.INTER_CUBIC,
        )

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(
            pafs,
            (0, 0),
            fx=upsample_ratio,
            fy=upsample_ratio,
            interpolation=cv2.INTER_CUBIC,
        )

        return heatmaps, pafs, scale, pad


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


if __name__ == "__main__":

    # Create a pose estimator
    pose_estimator = create_pose_estimator({"type": "py_openpose"})
    image = cv2.imread(str(DATA_DIR / "test_image.png"))

    face_detector = create_face_detector("retinaface")
    detections = face_detector.detect_faces(image)

    detections = pose_estimator.estimate_poses(image, detections)

    keypoints_annotator = KeyPointAnnotator(color=Color.red())
    image = keypoints_annotator.annotate(image, detections)

    plt.imshow(image)
    plt.show()
