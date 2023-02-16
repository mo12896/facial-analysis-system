# import os
# import sys
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np

from src.emotion.datahandler.dataprocessor.face_detector import create_face_detector
from src.emotion.datahandler.dataprocessor.head_pose_estimator import (
    create_head_pose_detector,
)
from src.emotion.utils.detections import Detections
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


@dataclass
class Identity:
    person_id: str
    n_vector: np.ndarray
    tip: np.ndarray
    points: np.ndarray
    sights: list[str] = field(default_factory=list)


class GazeDetector:
    def __init__(self, fov: int = 30, true_thresh: float = 0.8) -> None:
        self.fov = fov
        self.true_thresh = true_thresh

    @timer
    def detect_gazes(self, detections: Detections) -> Detections:
        """Function to detect gazes in a set of detections

        Args:
            detections (Detections): Detections object containing the detections

        Returns:
            Detections: Detection object with gaze detections
        """
        persons = detections.class_id
        poses = detections.head_pose_keypoints
        pts_res = detections.facial_landmarks

        identities: list[Identity] = []

        for person, (angles, translation), lmks in zip(persons, poses, pts_res):
            tip, n_vector, pts = self.prepare_data(
                angles[0],
                angles[1],
                angles[2],
                translation[0],
                translation[1],
                pts68=lmks,
            )
            identities.append(Identity(person, n_vector, tip, pts))

        for i in range(len(identities)):
            for j in range(len(identities)):
                pts_in_cone = self.detect_points_inside_cone(
                    identities[i].tip,
                    identities[i].n_vector,
                    1100,
                    identities[j].points,
                    identities[i].person_id,
                )
                if pts_in_cone.count(True) / len(pts_in_cone) >= self.true_thresh:
                    if not identities[i].person_id is identities[j].person_id:
                        identities[i].sights.append(identities[j].person_id)

        # for identity in identities:
        #     print(f"Person {identity.person_id} sees {identity.sights}")

        detections.gaze_detections = np.array(
            [identity.sights for identity in identities]
        )

        return detections

    @staticmethod
    def prepare_data(yaw, pitch, roll, tdx, tdy, size=1100, pts68=None) -> Tuple:
        """Function to prepare the data for the gaze detection

        Args:
            yaw (_type_): _description_
            pitch (_type_): _description_
            roll (_type_): _description_
            tdx (_type_): _description_
            tdy (_type_): _description_
            size (int, optional): _description_. Defaults to 1100.
            pts68 (_type_, optional): _description_. Defaults to None.

        Returns:
            Tuple: Tuple of cone tip, cone base and list of face associated points
        """
        pitch = pitch * np.pi / 180
        yaw = -(yaw * np.pi / 180)
        roll = roll * np.pi / 180

        if tdy > int(1053 / 2):

            if pts68 is not None:
                tdx = pts68[0][30]
                tdy = pts68[1][30]

                points = np.stack(
                    [pts68[0][:], pts68[1][:], -1 * np.array(range(68))], axis=1
                )

            # Z-Axis pointing out of the screen.
            x3 = size * (np.sin(yaw)) + tdx
            y3 = size * (-np.cos(yaw) * np.sin(pitch)) + tdy
            z3 = -size * (np.cos(pitch) * np.cos(yaw)) + 0

            vector = np.array([x3, y3, z3])

            tip = np.array([tdx, tdy, 0])

            return tip, vector, points

        else:
            # Half of image height
            y_offset = int(1053 / 2)
            # Estimated distance in between people in the room
            tdz = -1000

            if pts68 is not None:
                # Half of image width
                if tdx < int(1848 / 2):
                    tdx = pts68[0][30] + int(1848 / 2)
                    tdy = pts68[1][30] + y_offset

                    # Calculate the mean of the x-coordinates
                    x_mean = np.mean(pts68[0])

                    # Reflect the x-coordinates about the x-axis at their center point
                    x_reflected = -1 * (pts68[0] - x_mean) + x_mean

                    points = np.stack(
                        [
                            x_reflected + int(1848 / 2),
                            pts68[1][:] + y_offset,
                            1 * np.array(range(68)) + tdz,
                        ],
                        axis=1,
                    )
                else:
                    tdx = pts68[0][30] - int(1848 / 2)
                    tdy = pts68[1][30] + y_offset

                    # Calculate the mean of the x-coordinates
                    x_mean = np.mean(pts68[0])

                    # Reflect the x-coordinates about the x-axis at their center point
                    x_reflected = -1 * (pts68[0] - x_mean) + x_mean

                    points = np.stack(
                        [
                            x_reflected - int(1848 / 2),
                            pts68[1][:] + y_offset,
                            1 * np.array(range(68)) + tdz,
                        ],
                        axis=1,
                    )

            # Z-Axis pointing out of the screen. drawn in blue
            x3 = size * (np.sin(yaw + np.pi)) + tdx
            y3 = size * (-np.cos(yaw + np.pi) * np.sin(pitch)) + tdy
            z3 = -size * (np.cos(pitch) * np.cos(yaw + np.pi)) + tdz

            vector = np.array([x3, y3, z3])

            tip = np.array([tdx, tdy, tdz])

            return tip, vector, points

    def detect_points_inside_cone(
        self, tip_cone, base_cone, height, points: list, person=None
    ) -> List[bool]:
        """Detect if a list of points is inside a cone
        (https://stackoverflow.com/questions/12826117/how-can-i-detect-if-a-point-is-inside-a-cone-or-not-in-3d-space)

        Args:
            tip_cone (_type_): Coordinates of the tip of the cone
            base_cone (_type_): Normalized axis vector pointing from tip to base
            height (_type_): Height of cone
            radius (_type_): Base radius of the cone
            points (list): List point to test
        """

        points_inside_cone = []
        # Calculate the unit vector pointing in the direction of the cone
        vector = base_cone - tip_cone
        norm_vector = vector / np.linalg.norm(vector)
        # Based off assumption that the core binocular field of view of
        # humans is 60 degrees
        radius = np.tan(np.deg2rad(self.fov)) * height

        # Verification plots
        # ax.plot(
        #     [tip_cone[0], base_cone[0]],
        #     [tip_cone[1], base_cone[1]],
        #     [tip_cone[2], base_cone[2]],
        #     "b-",
        #     linewidth=2,
        # )

        # ax.text(
        #     tip_cone[0], tip_cone[1], tip_cone[2], person, color="black", fontsize=12, fontweight="bold"
        # )

        # ax.scatter(
        #     points[:, 0],
        #     points[:, 1],
        #     points[:, 2],
        #     color="black",
        #     s=1,
        # )

        # plot_truncated_cone(tip, base_cone, 0, np.tan(np.deg2rad(30)) * height)

        # calculate cone distance
        for p in points:
            # Calculate the vector from the tip of the cone to the given point
            dist = p - tip_cone
            # Calculate the distance from the tip of the cone to the given point
            # along the direction of the cone
            cone_dist = np.dot(dist, norm_vector)

            # reject points outside of the cone
            if cone_dist < 0 or cone_dist > height:
                points_inside_cone.append(False)
            else:
                # calculate cone radius and orthogonal distance
                cone_radius = (cone_dist / height) * radius
                orth_distance = np.linalg.norm((p - tip_cone) - cone_dist * norm_vector)

                # check if point is inside cone
                points_inside_cone.append(orth_distance < cone_radius)

        return points_inside_cone


if __name__ == "__main__":
    face_detector = create_face_detector({"type": "scrfd"})
    image = cv2.imread("/home/moritz/Workspace/masterthesis/data/test_image.png")
    detections = face_detector.detect_faces(image)

    head_pose_detector = create_head_pose_detector({"type": "synergy"})

    detections = head_pose_detector.detect_head_pose(image, detections)

    gaze_detector = GazeDetector()

    detections = gaze_detector.detect_gazes(detections)
