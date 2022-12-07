from dataclasses import dataclass, field
from typing import Callable

from dataclass_csv import DataclassReader, DataclassWriter

from .constants import IDENTITY_DIR


@dataclass(slots=True)
class IdentityState:
    """Identity state of the current user"""

    frame_count: int = 0
    confidence: float = 0.0
    bboxes: list = field(default_factory=list)
    emotions: list = field(default_factory=list)
    # TODO: add keypoints: Pose


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
        """Set the current state of the user

        Args:
            identity_state (IdentityState): The current state of the user
        """
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
                w.write()
                return

        with open(self.filename, "a") as f:
            w = DataclassWriter(f, [self._identity_state], IdentityState)
            w.write(skip_header=True)

    def read_states_from_csv(self) -> list[IdentityState]:
        """Convenience method to read user states from a csv file

        Returns:
            list[IdentityState]: List of identity states
        """
        identity_tracker = []

        with open(self.filename) as f:
            reader = DataclassReader(f, IdentityState)
            reader.map("Frame").to("frame_count")
            reader.map("Confidence").to("confidence")
            reader.map("Bounding Boxes").to("bboxes")
            reader.map("Emotions").to("emotions")

            for row in reader:
                row.bboxes = self.cast_list_of_strings(row.bboxes)
                row.emotions = self.cast_list_of_strings(row.emotions)
                identity_tracker.append(row)

            return identity_tracker

    @staticmethod
    def cast_list_of_strings(list_of_strings: list, dtype_dest: Callable = int) -> list:
        """Cast a list of strings to a list of desired dtypes

        Args:
            list_of_strings (list): List of strings to cast
            dtype_dest (Callable, optional): Desired datatype. Defaults to int.

        Returns:
            list: Final list of desired dtypes
        """
        return [
            dtype_dest(string)
            for string in list_of_strings
            if string not in ["[", "]", ",", " "]
        ]
