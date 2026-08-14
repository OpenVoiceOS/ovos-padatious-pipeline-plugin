"""Default confidence thresholds.

``conf_high`` defaults to the entity-hint identity boundary (0.9,
``pos_intent.ENTITY_HINT_IDENTITY``): out-of-list slot values blend to final
confidences in the low 0.94s, which straddled a 0.95 threshold
nondeterministically across training runs. The whole hint band must sit on
the high side, deterministically.
"""
import unittest

from unittest import mock

from ovos_padatious.opm import PadatiousPipeline
from ovos_padatious.pos_intent import ENTITY_HINT_IDENTITY


class TestDefaultThresholds(unittest.TestCase):
    def test_defaults(self):
        p = PadatiousPipeline(mock.Mock(), {"any": 1})  # non-empty: skip Configuration() fallback
        self.assertEqual(p.conf_high, ENTITY_HINT_IDENTITY)
        self.assertEqual(p.conf_high, 0.9)
        self.assertEqual(p.conf_med, 0.8)
        self.assertEqual(p.conf_low, 0.5)
        self.assertLess(p.conf_med, p.conf_high)
        self.assertLess(p.conf_low, p.conf_med)

    def test_config_override_wins(self):
        p = PadatiousPipeline(mock.Mock(), {"conf_high": 0.95})
        self.assertEqual(p.conf_high, 0.95)


if __name__ == "__main__":
    unittest.main()
