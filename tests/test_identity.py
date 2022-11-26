import unittest

from src.utils.identity import Identity, IdentityState


class IdentityTestCase(unittest.TestCase):
    def test_identity(self):
        identity_state = IdentityState()
        identity = Identity(identity_state)
        test_identity_state = IdentityState()
        identity.current_state = test_identity_state
        self.assertEqual(identity.current_state, identity_state)
