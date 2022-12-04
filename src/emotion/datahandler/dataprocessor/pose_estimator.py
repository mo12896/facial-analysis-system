from abc import ABC, abstractmethod

import torch

from emotion.models.body.light_openpose.load_state import load_state
from emotion.models.body.light_openpose.with_mobilenet import (
    PoseEstimationWithMobileNet,
)
from emotion.utils.constants import LIGHT_OPENPOSE_MODEL


class PoseEstimator(ABC):
    def __init__(self, model_path: str, cpu: str) -> None:
        self.model_path = model_path
        self.cpu = cpu

    @abstractmethod
    def construct_model(self) -> None:
        pass


def create_pose_estimator(estimator: str) -> PoseEstimator:
    """Factory method to create pose estimator objects."""
    if estimator == "light_openpose":
        return LightOpenPoseEstimator()
    else:
        raise NotImplementedError


class LightOpenPoseEstimator(PoseEstimator):
    def __init__(self, model_path=LIGHT_OPENPOSE_MODEL, cpu=False) -> None:
        super().__init__(model_path, cpu)

    def construct_model(self) -> PoseEstimationWithMobileNet:
        self.net = PoseEstimationWithMobileNet()
        checkpoint = torch.load(self.model_path, map_location="cpu")
        load_state(self.net, checkpoint)
        self.net.eval()
        if not self.cpu:
            self.net = self.net.cuda()
        return self.net
