"""T-1645: SimpleIntent.train re-seeds and retrains up to ten times. It must
keep the best attempt and stop when an attempt does not improve on it, not
run all ten and keep the last one."""
from unittest.mock import patch

from ovos_padatious import fann
from ovos_padatious.simple_intent import SimpleIntent
from ovos_padatious.train_data import TrainData


def _data():
    td = TrainData()
    td.add_lines("a", ["what is the weather", "how is the weather"])
    td.add_lines("b", ["set a timer", "start a timer"])
    return td


def _train_with_bit_fails(fails):
    """Train intent 'a' with train_on_data stubbed so attempt i reports
    fails[i] failing rows. Returns (attempts made, bit_fail of the kept net)."""
    calls = []

    def fake_train(self, data, max_epochs, ebr, desired):
        calls.append(self)

    def fake_test(self, data):
        self.bit_fail = fails[calls.index(self)]

    with patch.object(fann.neural_net, "train_on_data", fake_train), \
            patch.object(fann.neural_net, "test_data", fake_test):
        intent = SimpleIntent("a")
        intent.train(_data())
    return len(calls), intent.net.get_bit_fail(), calls.index(intent.net)


def test_stops_on_first_clean_attempt():
    attempts, kept, idx = _train_with_bit_fails([5, 0, 3])
    assert attempts == 2 and kept == 0 and idx == 1


def test_keeps_the_best_attempt_not_the_last():
    # fail-before: dev ran all ten and kept attempt 10 (bit_fail 9)
    attempts, kept, idx = _train_with_bit_fails([7, 4, 6, 8, 5, 3, 9, 9, 9, 9])
    assert kept == 4 and idx == 1
    assert attempts == 3  # attempt 3 did not improve on attempt 2


def test_a_later_improvement_is_kept_until_the_first_non_improvement():
    attempts, kept, idx = _train_with_bit_fails([9, 6, 2, 2, 1])
    assert attempts == 4 and kept == 2 and idx == 2
