from pathlib import Path

from src.emotion.utils.identity import IdentityHandler, IdentityState

IDENTITY_PATH = "/home/moritz/Workspace/masterthesis/data/identity/test.csv"


def test_identity_state():
    """Test identity state"""
    identity_state = IdentityState()
    assert identity_state.frame_count == 0
    assert identity_state.confidence == 0.0
    assert identity_state.bboxes == []
    assert identity_state.face == []
    assert identity_state.emotions == []


def test_identity_handler():
    """Test identity handler"""
    identity_handler = IdentityHandler("test")
    assert identity_handler.user_id == "test"
    assert identity_handler.filename == Path(IDENTITY_PATH)
    assert identity_handler.identity_state == IdentityState()
