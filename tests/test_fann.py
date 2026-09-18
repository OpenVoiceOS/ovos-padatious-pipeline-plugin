"""Tests for the numpy neural network backend (ovos_padatious.fann).

The fixture ``.net`` files under tests/fixtures were trained and saved with
genuine libfann (via fann2); the paired ``.json`` files hold probe input
vectors and the outputs libfann produced for them. They pin the FANN_FLO_2.1
reader and the forward pass to libfann's numeric behavior.
"""
import json
import os
import unittest

from ovos_padatious import fann

FIXTURES = os.path.join(os.path.dirname(__file__), 'fixtures')


def load_fixture(name):
    net = fann.neural_net()
    assert net.create_from_file(os.path.join(FIXTURES, name + '.net'))
    with open(os.path.join(FIXTURES, name + '.json')) as f:
        probes = json.load(f)
    return net, probes


class TestLibfannParity(unittest.TestCase):
    def check_parity(self, name):
        net, probes = load_fixture(name)
        for vec, expected in zip(probes['probes'], probes['outputs']):
            self.assertAlmostEqual(net.run(vec)[0], expected[0], places=5)

    def test_symmetric_net_matches_libfann(self):
        # intent-shaped net: hidden+output SIGMOID_SYMMETRIC_STEPWISE
        self.check_parity('intent_sym')

    def test_asymmetric_output_matches_libfann(self):
        # entity-edge-shaped net: output SIGMOID_STEPWISE
        self.check_parity('edge_asym')

    def test_loaded_config(self):
        net, _ = load_fixture('intent_sym')
        self.assertEqual(net.layers, [8, 10, 1])
        self.assertEqual(net.hidden_activation, fann.SIGMOID_SYMMETRIC_STEPWISE)
        self.assertEqual(net.output_activation, fann.SIGMOID_SYMMETRIC_STEPWISE)
        self.assertAlmostEqual(net.bit_fail_limit, 0.1)
        self.assertAlmostEqual(net.steepness, 0.5)


class TestPersistence(unittest.TestCase):
    def test_save_load_roundtrip_is_exact(self):
        for name in ('intent_sym', 'edge_asym'):
            net, probes = load_fixture(name)
            path = os.path.join(FIXTURES, 'roundtrip_tmp.net')
            try:
                net.save(path)
                net2 = fann.neural_net()
                self.assertTrue(net2.create_from_file(path))
                for vec in probes['probes']:
                    self.assertEqual(net2.run(vec)[0], net.run(vec)[0])
            finally:
                if os.path.exists(path):
                    os.remove(path)

    def test_load_missing_file_returns_false(self):
        net = fann.neural_net()
        self.assertFalse(net.create_from_file('/no/such/file.net'))

    def test_load_garbage_returns_false(self):
        path = os.path.join(FIXTURES, 'garbage_tmp.net')
        with open(path, 'w') as f:
            f.write('not a fann file\n')
        try:
            self.assertFalse(fann.neural_net().create_from_file(path))
        finally:
            os.remove(path)


class TestTraining(unittest.TestCase):
    def _train(self, output_activation, low):
        inputs = [[float(i == j) for j in range(6)] for i in range(6)]
        outputs = [[1.0 if i < 3 else low] for i in range(6)]
        data = fann.training_data()
        data.set_train_data(inputs, outputs)
        for _ in range(10):
            net = fann.neural_net()
            net.create_standard_array([6, 3, 1])
            net.set_activation_function_hidden(fann.SIGMOID_SYMMETRIC_STEPWISE)
            net.set_activation_function_output(output_activation)
            net.set_train_stop_function(fann.STOPFUNC_BIT)
            net.set_bit_fail_limit(0.1)
            net.train_on_data(data, 1000, 0, 0)
            net.test_data(data)
            if net.get_bit_fail() == 0:
                break
        return net, inputs, outputs

    def test_trains_symmetric_to_zero_bit_fail(self):
        net, inputs, outputs = self._train(fann.SIGMOID_SYMMETRIC_STEPWISE, -1.0)
        self.assertEqual(net.get_bit_fail(), 0)
        for vec, target in zip(inputs, outputs):
            self.assertAlmostEqual(net.run(vec)[0], target[0], delta=0.1)

    def test_trains_asymmetric_to_zero_bit_fail(self):
        net, inputs, outputs = self._train(fann.SIGMOID_STEPWISE, 0.0)
        self.assertEqual(net.get_bit_fail(), 0)
        for vec, target in zip(inputs, outputs):
            self.assertAlmostEqual(net.run(vec)[0], target[0], delta=0.1)

    def test_bit_fail_counts_failures(self):
        net = fann.neural_net()
        net.create_standard_array([2, 3, 1])
        net.set_bit_fail_limit(0.1)
        data = fann.training_data()
        data.set_train_data([[0.0, 0.0], [1.0, 1.0]], [[1.0], [-1.0]])
        net.test_data(data)  # untrained tiny weights -> output near 0
        self.assertEqual(net.get_bit_fail(), 2)

    def test_mismatched_data_raises(self):
        data = fann.training_data()
        with self.assertRaises(ValueError):
            data.set_train_data([[0.0, 1.0]], [[1.0], [0.0]])


if __name__ == '__main__':
    unittest.main()
