import cv2

from external.synergy.utils.inference import draw_axis
from src.emotion.utils.detections import Detections


class HeadPoseAnnotator:
    def __init__(self, parameters: dict = {}):
        """Base constructor for all head pose detectors.

        Args:
            parameters (dict, optional): Parameters for the head pose detector. Defaults to {}.
        """

        self.parameters = parameters

    def annotate(self, image_cpy, detections: Detections):

        pts_res = detections.facial_landmarks
        poses = detections.head_pose_keypoints

        image_cpy = self.draw_landmarks(image_cpy, pts_res)

        for angles, translation, lmks in poses:
            image_cpy = draw_axis(
                image_cpy,
                angles[0],
                angles[1],
                angles[2],
                translation[0],
                translation[1],
                size=50,
                pts68=lmks,
            )

        return image_cpy

    @staticmethod
    def draw_landmarks(img, pts: list):
        markersize = 1
        color = (255, 255, 255)
        nums = [0, 17, 22, 27, 31, 36, 42, 48, 60, 68]

        for i in range(len(pts)):
            for ind in range(len(nums) - 1):
                l, r = nums[ind], nums[ind + 1]
                for j in range(l, r):
                    cv2.circle(
                        img,
                        (int(pts[i][0, j]), int(pts[i][1, j])),
                        markersize,
                        color,
                        -1,
                    )
                    if j < r - 1:
                        cv2.line(
                            img,
                            (int(pts[i][0, j]), int(pts[i][1, j])),
                            (int(pts[i][0, j + 1]), int(pts[i][1, j + 1])),
                            color,
                            1,
                        )

        return img
