from dataclasses import dataclass

import numpy as np
from constants import IDENTITY_DIR
from dataclass_csv import DataclassWriter


@dataclass(slots=True)
class IdentityState:
    frame_count: int = 0
    confidence: float = 0.0
    bboxes: np.ndarray = np.array([])
    face: np.ndarray = np.array([])
    emotions: list = []
    # keypoints: Pose


class IdentityHandler:
    """Identity handler for the current users state"""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.filename = IDENTITY_DIR / f"{self.user_id}.csv"
        self.identity_state: IdentityState = IdentityState()

    @property
    def current_state(self) -> IdentityState:
        return self._identity_state

    @current_state.setter
    def current_state(self, identity_state: IdentityState):
        try:
            self.write_state_to_csv()
        except Exception as exc:
            print(exc)

        self._identity_state = identity_state

    def write_state_to_csv(self) -> None:
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
                return

        with open(self.filename, "w") as f:
            w = DataclassWriter(f, [self.identity_state], IdentityState)
            w.write(skip_header=True)
