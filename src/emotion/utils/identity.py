from dataclasses import dataclass
from typing import Callable

from dataclass_csv import DataclassReader, DataclassWriter

from .constants import IDENTITY_DIR
from .detections import Detections


@dataclass(slots=True)
class IdentityState:
    """Identity state of the current user"""

    frame_count: int = 0
    class_id: str = ""
    confidence: float = 0.0
    xmin: int = 0
    ymin: int = 0
    xmax: int = 0
    ymax: int = 0
    angry: float = 0.0
    disgust: float = 0.0
    happy: float = 0.0
    sad: float = 0.0
    surprise: float = 0.0
    fear: float = 0.0
    neutral: float = 0.0


class IdentityHandler:
    """Identity handler for the current users state"""

    def __init__(self):
        self.filename = IDENTITY_DIR / "identities.csv"
        self._identities_states: list[IdentityState] = []

    def __len__(self):
        return len(self._identities_states)

    def get_current_state(self) -> list[IdentityState]:
        return self._identities_states

    def set_current_state(self, detections: Detections, frame: int):
        """Set the current state of the user

        Args:
            identity_state (IdentityState): The current state of the user
        """
        self.parse_detections(detections, frame)

    def parse_detections(self, detections: Detections, frame: int) -> None:
        """Parse the detections to a user state

        Args:
            detections (Detections): Detections object

        Returns:
            IdentityState: User state
        """
        self._identities_states = []

        for bbox, conf, class_id, _, _, emotion, _ in detections:
            if emotion is None:
                emotion = {}

            identity_state = IdentityState(
                frame_count=frame,
                class_id=class_id,
                confidence=conf,
                xmin=bbox[0],
                ymin=bbox[1],
                xmax=bbox[2],
                ymax=bbox[3],
                angry=emotion.get("angry", 0.0),
                disgust=emotion.get("disgust", 0.0),
                happy=emotion.get("happy", 0.0),
                sad=emotion.get("sad", 0.0),
                surprise=emotion.get("surprise", 0.0),
                fear=emotion.get("fear", 0.0),
                neutral=emotion.get("neutral", 0.0),
            )
            self._identities_states.append(identity_state)

    def write_states_to_csv(self) -> None:
        for identity_state in self._identities_states:
            if not self.filename.exists():
                self.filename.touch()

                with open(self.filename, "w") as f:
                    w = DataclassWriter(f, [identity_state], IdentityState)
                    w.map("frame_count").to("Frame")
                    w.map("confidence").to("Confidence")
                    w.map("class_id").to("ClassID")
                    w.map("xmin").to("XMin")
                    w.map("ymin").to("YMin")
                    w.map("xmax").to("XMax")
                    w.map("ymax").to("YMax")
                    w.map("angry").to("Angry")
                    w.map("disgust").to("Disgust")
                    w.map("happy").to("Happy")
                    w.map("sad").to("Sad")
                    w.map("surprise").to("Surprise")
                    w.map("fear").to("Fear")
                    w.map("neutral").to("Neutral")
                    w.write()
                    return

            with open(self.filename, "a") as f:
                w = DataclassWriter(f, [identity_state], IdentityState)
                w.write(skip_header=True)

    # TODO: Not tested!
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
            reader.map("ClassID").to("class_id")
            reader.map("XMin").to("xmin")
            reader.map("YMin").to("ymin")
            reader.map("XMax").to("xmax")
            reader.map("YMax").to("ymax")
            reader.map("Angry").to("angry")
            reader.map("Disgust").to("disgust")
            reader.map("Happy").to("happy")
            reader.map("Sad").to("sad")
            reader.map("Surprise").to("surprise")
            reader.map("Fear").to("fear")
            reader.map("Neutral").to("neutral")

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
