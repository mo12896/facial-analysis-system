from dataclasses import dataclass
from typing import Tuple

import numpy as np
from constants import DATABASE_DIR, IDENTITY_DIR
from dataclass_csv import DataclassWriter


@dataclass(slots=True)
class IdentityState:
    frame_count: int = 0
    confidence: float = 0.0
    bboxes: np.ndarray = np.array([])
    face: np.ndarray = np.array([])
    emotions: list = []
    # keypoints: Pose


class Identity:
    """Identity class for the user."""

    def __init__(self, identity_state: IdentityState = IdentityState()):
        self.identity_state = identity_state

    @property
    def current_state(self) -> IdentityState:
        return self._identity_state

    @current_state.setter
    def current_state(self, identity_state: IdentityState):
        confidence = self.verify_identity(identity_state.face)

        identity_state.confidence = confidence
        with open(self.filename, "w") as f:
            w = DataclassWriter(f, [identity_state], IdentityState)
            w.write(skip_header=True)

        self._identity_state = identity_state

    def verify_identity(self, frame: np.ndarray) -> bool | float:
        user_id, confidence = self.verification_process(frame)

        if not confidence:
            # TODO: Define verification handling
            return False

        self.filename = IDENTITY_DIR / f"{user_id}.csv"

        if not self.filename.exists():
            self.filename.touch()
            with open(self.filename, "w") as f:
                w = DataclassWriter(f, [self.identity_state], IdentityState)
                w.map("frame_count").to("Frame")
                w.map("confidence").to("Confidence")
                w.map("bboxes").to("Bounding Boxes")
                w.map("face").to("Face")
                w.map("emotions").to("Emotions")
                # w.map("keypoints").to("Keypoints")
        return confidence

    # TODO: Define a verficiation logic
    @staticmethod
    def verification_process(frame: np.ndarray) -> Tuple[int, float]:
        pass
