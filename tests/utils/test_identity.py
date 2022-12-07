from pathlib import Path

import pytest
from dataclass_csv import DataclassReader

from src.emotion.utils.helper_functions import cast_list_of_strings
from src.emotion.utils.identity import IdentityHandler, IdentityState

IDENTITY_PATH = "/home/moritz/Workspace/masterthesis/data/identities/test.csv"

test_identities: dict[str, IdentityState] = {
    "id_1": IdentityState(
        frame_count=1,
        confidence=0.5,
        bboxes=[1, 2, 3, 4],
        emotions=[1, 2, 3, 4, 5, 6, 7],
    ),
    "id_2": IdentityState(
        frame_count=2,
        confidence=0.5,
        bboxes=[1, 2, 3, 4],
        emotions=[1, 2, 3, 4, 5, 6, 7],
    ),
}


def test_identity_state():
    """Test identity state"""
    identity_state = IdentityState()
    assert identity_state.frame_count == 0
    assert identity_state.confidence == 0.0
    assert identity_state.bboxes == []
    assert identity_state.emotions == []


@pytest.fixture()
def identity():
    """Setup identity"""
    identity = IdentityHandler("test")
    yield identity


class TestIdentityHandler:
    """Test identity handler with pytest"""

    def test_identity_handler(self, identity: IdentityHandler):
        """Test identity handler"""
        assert identity.user_id == "test"
        assert identity.filename == Path(IDENTITY_PATH)
        assert identity.current_state == IdentityState()

    def test_false_identity_handler(self):
        """Test false identity"""
        with pytest.raises(ValueError):
            IdentityHandler("test/test")

    def test_write_state_to_csv(
        self,
        identity: IdentityHandler,
        test_identity: IdentityState = test_identities["id_1"],
    ):
        """Test write state to csv"""
        identity.current_state = test_identity
        with open(identity.filename) as identity_csv:
            reader = DataclassReader(identity_csv, IdentityState)
            reader.map("Frame").to("frame_count")
            reader.map("Confidence").to("confidence")
            reader.map("Bounding Boxes").to("bboxes")
            reader.map("Emotions").to("emotions")
            for row in reader:
                assert row.frame_count == test_identity.frame_count
                assert row.confidence == test_identity.confidence
                assert cast_list_of_strings(row.bboxes) == test_identity.bboxes
                assert cast_list_of_strings(row.emotions) == test_identity.emotions
        identity.filename.unlink()

    def test_write_state_to_csv_with_existing_file(
        self,
        identity: IdentityHandler,
        test_ids: dict[str, IdentityState] = test_identities,
    ):
        """Test write state to csv with existing file"""
        for test_id in test_ids:
            identity.current_state = test_ids[test_id]

        with open(identity.filename) as identity_csv:
            reader = DataclassReader(identity_csv, IdentityState)
            reader.map("Frame").to("frame_count")
            reader.map("Confidence").to("confidence")
            reader.map("Bounding Boxes").to("bboxes")
            reader.map("Emotions").to("emotions")
            for count, row in enumerate(reader):
                assert row.frame_count == test_ids[f"id_{count+1}"].frame_count
                assert row.confidence == test_ids[f"id_{count+1}"].confidence
                assert (
                    cast_list_of_strings(row.bboxes) == test_ids[f"id_{count+1}"].bboxes
                )
                assert (
                    cast_list_of_strings(row.emotions)
                    == test_ids[f"id_{count+1}"].emotions
                )
        identity.filename.unlink()
