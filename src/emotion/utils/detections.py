from typing import Dict, List, Optional

import numpy as np


class Detections:
    def __init__(
        self,
        bboxes: np.ndarray,
        confidence: np.ndarray,
        class_id: np.ndarray,
        bbox_centerpoints: Optional[np.ndarray] = None,
        body_pose_keypoints: Optional[np.ndarray] = None,
        facial_landmarks: Optional[np.ndarray] = None,
        head_pose_keypoints: Optional[List] = None,
        emotion: Optional[np.ndarray] = None,
        tracker_id: Optional[np.ndarray] = None,
    ):
        self.bboxes = bboxes
        self.confidence = confidence
        self.class_id = class_id
        self.bbox_centerpoints = self.compute_bbox_center_points(self.bboxes)
        self.body_pose_keypoints = body_pose_keypoints
        self.facial_landmarks = facial_landmarks
        self.head_pose_keypoints = head_pose_keypoints
        self.emotion = emotion
        self.tracker_id = tracker_id

        n = len(self.bboxes)
        validators = [
            (isinstance(self.bboxes, np.ndarray) and self.bboxes.shape == (n, 4)),
            (isinstance(self.confidence, np.ndarray) and self.confidence.shape == (n,)),
            (isinstance(self.class_id, np.ndarray) and self.class_id.shape == (n,)),
            self.emotion is None
            or (isinstance(self.emotion, np.ndarray) and self.emotion.shape == (n,)),
            self.tracker_id is None
            or (
                isinstance(self.tracker_id, np.ndarray)
                and self.tracker_id.shape == (n,)
            ),
        ]

        if not all(validators):
            raise ValueError(
                "bboxes must be 2d np.ndarray with (n, 4) shape, "
                "confidence must be 1d np.ndarray with (n,) shape, "
                "class_id must be 1d np.ndarray with (n,) shape, "
                "tracker_id must be None or 1d np.ndarray with (n,) shape"
            )

    def __len__(self):
        return len(self.bboxes)

    def __iter__(self):
        """Iterate over detections and yield a tuple of (bboxes, confidence, class_id, tracker_id)"""
        for i in range(len(self.bboxes)):
            yield (
                self.bboxes[i],
                self.confidence[i],
                self.class_id[i],
                self.bbox_centerpoints[i]
                if self.bbox_centerpoints is not None
                else None,
                self.body_pose_keypoints[i]
                if self.body_pose_keypoints is not None
                else None,
                self.facial_landmarks[i] if self.facial_landmarks is not None else None,
                self.head_pose_keypoints[i]
                if self.head_pose_keypoints is not None
                else None,
                self.emotion[i] if self.emotion is not None else None,
                self.tracker_id[i] if self.tracker_id is not None else None,
            )

    # Legacy implementation from retinface package (15x slower than insightface)
    @classmethod
    def from_retinaface(cls, retinaface_output: dict):
        """Create Detections object from RetinaFace output"""

        bboxes = []
        confidence = []
        class_id = []

        for face in retinaface_output:
            bboxes.append(retinaface_output[face]["facial_area"])
            confidence.append(retinaface_output[face]["score"])
            class_id.append(face)

        # Note that setting the dtype for class_id is important to keep the final output strings!
        return cls(
            np.array(bboxes),
            np.array(confidence),
            np.array(class_id, dtype="U10"),
        )

    @classmethod
    def from_scrfd(cls, scrfd_output: List[Dict]):
        """Create Detections object from RetinaFace output"""

        bboxes = []
        confidence = []
        class_id = []

        for i in range(len(scrfd_output)):
            bboxes.append(scrfd_output[i]["bbox"])
            confidence.append(scrfd_output[i]["det_score"])
            class_id.append(f"face_{i}")

        # Note that setting the dtype for class_id is important to keep the final output strings!
        return cls(
            np.array(bboxes),
            np.array(confidence),
            np.array(class_id, dtype="U10"),
        )

    @classmethod
    def from_mediapipe(cls, mediapipe_output: dict, image_size):
        """Create Detections object from MediaPipe output"""

        bboxes = []
        confidence = []
        class_id = []
        image_height, image_width = image_size
        lines = str(mediapipe_output).strip().split(",")

        # Bulky parser, because mediapipe...
        for line in lines:
            parts = line.strip().split("\n")
            label_id = int(parts[0].split(":")[1].strip())
            score = float(parts[1].split(":")[1].strip())
            xmin = float(parts[5].split(":")[1].strip())
            ymin = float(parts[6].split(":")[1].strip())
            width = float(parts[7].split(":")[1].strip())
            height = float(parts[8].strip().split(":")[1].strip())

            xmin = int(xmin * image_width)
            ymin = int(ymin * image_height)
            width = int(width * image_width)
            height = int(height * image_height)

            bbox = np.array([xmin, ymin, xmin + width, ymin + height])

            bboxes.append(bbox)
            confidence.append(score)
            class_id.append(label_id)

        # Note that setting the dtype for class_id is important to keep the final output strings!
        return cls(
            np.array(bboxes), np.array(confidence), np.array(class_id, dtype="U10")
        )

    def filter(self, mask: np.ndarray, inplace: bool = False):
        """Filter detections by mask

        Args:
            mask (np.ndarray): Mask of shape (n,) containing boolean values to filter detections
            inplace (bool, optional): If True, filter detections inplace. Defaults to False.

        """
        if inplace:
            self.bboxes = self.bboxes[mask]
            self.confidence = self.confidence[mask]
            self.class_id = self.class_id[mask]
            self.emotion = self.emotion[mask] if self.emotion is not None else None
            self.tracker_id = (
                self.tracker_id[mask] if self.tracker_id is not None else None
            )
            return self
        else:
            return Detections(
                bboxes=self.bboxes[mask],
                confidence=self.confidence[mask],
                class_id=self.class_id[mask],
                bbox_centerpoints=self.bbox_centerpoints[mask]
                if self.bbox_centerpoints is not None
                else None,
                emotion=self.emotion[mask] if self.emotion is not None else None,
                tracker_id=self.tracker_id[mask]
                if self.tracker_id is not None
                else None,
            )

    def match_poses(self, group_keypoints):
        body_pose_keypoints = np.zeros(
            (len(self.bbox_centerpoints), 18, 2), dtype=np.float32
        )

        for bbox_center in range(len(self.bbox_centerpoints)):
            smallest_distance = 100000
            for person in range(len(group_keypoints)):
                distance = np.linalg.norm(
                    self.bbox_centerpoints[bbox_center] - group_keypoints[person][0]
                )
                if distance < smallest_distance:
                    smallest_distance = distance
                    body_pose_keypoints[bbox_center] = group_keypoints[person]

        return Detections(
            bboxes=self.bboxes,
            confidence=self.confidence,
            class_id=self.class_id,
            bbox_centerpoints=self.bbox_centerpoints,
            body_pose_keypoints=body_pose_keypoints,
            emotion=self.emotion,
            tracker_id=self.tracker_id,
        )

    def match_head_poses(self, poses, pts_res):
        facial_landmarks = np.zeros(
            (len(self.bbox_centerpoints), 3, 68), dtype=np.float32
        )
        head_pose_keypoints = [0] * len(self.bbox_centerpoints)
        for bbox_center in range(len(self.bbox_centerpoints)):
            smallest_distance = 100000
            for person in range(len(poses)):
                # Note that 30 is the noise keypoint!
                distance = np.linalg.norm(
                    self.bbox_centerpoints[bbox_center]
                    - (int(pts_res[person][0, 30]), int(pts_res[person][1, 30]))
                )
                if distance < smallest_distance:
                    smallest_distance = distance
                    facial_landmarks[bbox_center] = pts_res[person]
                    head_pose_keypoints[bbox_center] = poses[person]

        return Detections(
            bboxes=self.bboxes,
            confidence=self.confidence,
            class_id=self.class_id,
            bbox_centerpoints=self.bbox_centerpoints,
            facial_landmarks=facial_landmarks,
            head_pose_keypoints=head_pose_keypoints,
            emotion=self.emotion,
            tracker_id=self.tracker_id,
        )

    @classmethod
    def compute_bbox_center_points(cls, bboxes: np.ndarray):
        """Compute center points of bounding boxes

        Args:
            bboxes (np.ndarray): Bounding boxes of shape (n, 4)

        Returns:
            np.ndarray: Center points of shape (n, 2)
        """
        return np.array([cls.compute_center_point(bbox) for bbox in bboxes])

    @staticmethod
    def compute_center_point(bbox):
        """Compute center point of a bounding box

        Args:
            bbox (np.ndarray): Bounding box of shape (4,)

        Returns:
            np.ndarray: Center point of shape (2,)
        """
        return np.array(
            [
                (bbox[0] + bbox[2]) / 2,
                (bbox[1] + bbox[3]) / 2,
            ]
        )
