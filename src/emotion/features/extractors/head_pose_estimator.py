# import os
# import sys
from abc import ABC, abstractmethod

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from synergy3DMM import SynergyNet

from external.synergy.utils.ddfa import Normalize, ToTensor
from external.synergy.utils.inference import crop_img, predict_pose, predict_sparseVert
from src.emotion.features.annotators.head_annotator import HeadPoseAnnotator
from src.emotion.features.detections import Detections
from src.emotion.features.extractors.face_detector import create_face_detector
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


class HeadPoseDetector(ABC):
    @abstractmethod
    def __init__(self, parameters: dict = {}):
        """Base constructor for all head pose detectors.

        Args:
            parameters (dict, optional): Parameters for the head pose detector. Defaults to {}.
        """

        self.parameters = parameters

    @abstractmethod
    def detect_head_poses(
        self, image: np.ndarray, detections: Detections
    ) -> Detections:
        """Abstract method to detect head pose in a given frame.

        Args:
            frame (np.ndarray): Current frame

        Returns:
            Detections: Object which holds the bounding boxes, confidences, and class ids
        """


def create_head_pose_detector(parameters: dict) -> HeadPoseDetector:
    """Factory method to create head pose detector objects.

    Args:
        parameters (dict): Parameters for the head pose detector

    Returns:
        HeadPoseDetector: Head pose detector object
    """

    if parameters["type"] == "synergy":
        return SynergyHeadPoseDetector(parameters)

    raise ValueError("Invalid head pose detector")


class SynergyHeadPoseDetector(HeadPoseDetector):
    def __init__(self, parameters: dict = {}):
        super().__init__(parameters)
        self.model = SynergyNet()
        self.gpu = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.gpu else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.img_size = 120

    @timer
    def detect_head_poses(
        self, image: np.ndarray, detections: Detections
    ) -> Detections:
        pts_res = []
        poses = []

        for bbox in detections.bboxes:

            transform = transforms.Compose([ToTensor(), Normalize(mean=127.5, std=128)])
            roi_box = np.append(bbox, [1])

            # enlarge the bbox a little and do a square crop
            HCenter = (roi_box[1] + roi_box[3]) / 2
            WCenter = (roi_box[0] + roi_box[2]) / 2
            side_len = roi_box[3] - roi_box[1]
            margin = side_len * 1.2 // 2
            roi_box[0], roi_box[1], roi_box[2], roi_box[3] = (
                WCenter - margin,
                HCenter - margin,
                WCenter + margin,
                HCenter + margin,
            )

            img = crop_img(image, roi_box)
            img = cv2.resize(
                img,
                dsize=(self.img_size, self.img_size),
                interpolation=cv2.INTER_LINEAR,
            )

            input = transform(img).unsqueeze(0)
            with torch.no_grad():
                if self.gpu:
                    input = input.cuda()
                param = self.model.forward_test(input)
                param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

            # inferences
            lmks = predict_sparseVert(param, roi_box, transform=True)
            angles, translation = predict_pose(param, roi_box)

            pts_res.append(lmks)
            poses.append([angles, translation])

        detections = detections.match_head_poses(poses, pts_res)

        return detections


if __name__ == "__main__":
    face_detector = create_face_detector({"type": "scrfd"})
    image = cv2.imread("/home/moritz/Workspace/masterthesis/data/test_image.png")
    detections = face_detector.detect_faces(image)

    head_pose_detector = create_head_pose_detector({"type": "synergy"})

    detections = head_pose_detector.detect_head_poses(image, detections)

    annotator = HeadPoseAnnotator()
    img = annotator.annotate(image, detections)

    plt.imshow(img)
    plt.show()
