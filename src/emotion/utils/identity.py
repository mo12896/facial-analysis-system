from dataclasses import dataclass, field

from dataclass_csv import DataclassWriter

from .constants import IDENTITY_DIR


@dataclass(slots=True)
class IdentityState:
    frame_count: int = 0
    confidence: float = 0.0
    bboxes: list = field(default_factory=list)
    emotions: list = field(default_factory=list)
    # keypoints: Pose


class IdentityHandler:
    """Identity handler for the current users state"""

    def __init__(self, user_id: str):
        if any([char in user_id for char in ["/"]]):
            raise ValueError("User ID cannot contain '/'")
        self.user_id = user_id
        self.filename = IDENTITY_DIR / f"{self.user_id}.csv"
        self._identity_state: IdentityState = IdentityState()

    @property
    def current_state(self) -> IdentityState:
        return self._identity_state

    @current_state.setter
    def current_state(self, identity_state: IdentityState):
        self._identity_state = identity_state
        self._write_state_to_csv()

    def _write_state_to_csv(self) -> None:
        if not self.filename.exists():
            self.filename.touch()

            with open(self.filename, "w") as f:
                w = DataclassWriter(f, [self._identity_state], IdentityState)
                w.map("frame_count").to("Frame")
                w.map("confidence").to("Confidence")
                w.map("bboxes").to("Bounding Boxes")
                w.map("emotions").to("Emotions")
                # w.map("keypoints").to("Keypoints")
                w.write()
                return

        with open(self.filename, "w") as f:
            w = DataclassWriter(f, [self._identity_state], IdentityState)
            w.write(skip_header=True)
