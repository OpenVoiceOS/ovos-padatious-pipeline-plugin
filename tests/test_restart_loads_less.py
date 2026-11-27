"""Restarting with unchanged skills loads and tokenizes less."""
import shutil
import tempfile
import unittest
from unittest import mock

from ovos_bus_client.message import Message
from ovos_utils.messagebus import FakeBus
from xxhash import xxh32

from ovos_padatious.intent import Intent
from ovos_padatious.intent_container import IntentContainer
from ovos_padatious.opm import PadatiousPipeline
from ovos_padatious.util import lines_hash

SKILL_ID = "once.skill"
LANG = "en-US"
SAMPLES = ["what time is it", "tell me the time", "time (please|now)"]


def legacy_msg(samples=SAMPLES):
    return Message("padatious:register_intent", {
        "skill_id": SKILL_ID, "name": f"{SKILL_ID}:time.intent", "lang": LANG,
        "samples": list(samples),
    }, {"skill_id": SKILL_ID})


def spec_msg(samples=SAMPLES):
    return Message("ovos.intent.register.template", {
        "skill_id": SKILL_ID, "intent_name": "time", "lang": LANG,
        "samples": list(samples), "blacklist": [],
    }, {"skill_id": SKILL_ID})


class _Pipeline(unittest.TestCase):
    def setUp(self):
        self.cache = tempfile.mkdtemp()
        self.pipeline = PadatiousPipeline(FakeBus(), config={"intent_cache": self.cache})

    def tearDown(self):
        self.pipeline.shutdown()
        shutil.rmtree(self.cache, ignore_errors=True)

    def settle(self):
        self.assertTrue(self.pipeline.wait_until_trained(timeout=30))


class TestRestartKeepsLoadedModel(_Pipeline):
    def test_identical_hash_reuses_the_live_model(self):
        container = self.pipeline.containers[LANG]
        container.add_intent("x:time", SAMPLES)
        container.train()
        live = next(i for i in container.intents.objects if i.name == "x:time")
        with mock.patch.object(Intent, "from_file", wraps=Intent.from_file) as load:
            container.add_intent("x:time", SAMPLES)
        self.assertEqual(load.call_count, 0,
                         "an unchanged intent was reloaded from the cache")
        self.assertIs(next(i for i in container.intents.objects
                           if i.name == "x:time"), live)
        self.assertFalse(container.must_train)
    def test_a_fresh_container_still_loads_from_the_cache(self):
        container = self.pipeline.containers[LANG]
        container.add_intent("x:time", SAMPLES)
        container.train()
        fresh = IntentContainer(self.cache + "/" + LANG)
        with mock.patch.object(Intent, "from_file", wraps=Intent.from_file) as load:
            fresh.add_intent("x:time", SAMPLES)
        self.assertEqual(load.call_count, 1)
        self.assertFalse(fresh.must_train)
        fresh.shutdown()


class TestLinesHashIsStable(unittest.TestCase):
    def test_digest_equals_the_per_line_stream(self):
        # every intent cache on disk is keyed on this digest
        lines = ["2.2", "format2", "b line", "a line", "ünïcode", ""]
        x = xxh32()
        for line in sorted(lines):
            x.update(line.encode())
        self.assertEqual(lines_hash(lines), x.digest())


class TestTokenizedOnlyForTraining(unittest.TestCase):
    def test_cache_hit_tokenizes_nothing(self):
        cache = tempfile.mkdtemp()
        try:
            first = IntentContainer(cache)
            first.add_intent("x:time", SAMPLES)
            first.train()
            first.shutdown()
            fresh = IntentContainer(cache)
            with mock.patch("ovos_padatious.train_data.tokenize") as tok:
                fresh.add_intent("x:time", SAMPLES)
                fresh.train()
            tok.assert_not_called()
            self.assertEqual(fresh.calc_intent("what time is it").name, "x:time")
            fresh.shutdown()
        finally:
            shutil.rmtree(cache, ignore_errors=True)
    def test_training_still_reads_every_sentence(self):
        from ovos_padatious.train_data import TrainData
        data = TrainData()
        data.add_lines("a", ["hello there", "hi"])
        data.add_lines("b", ["bye now"])
        self.assertEqual(sorted(map(tuple, data.my_sents("a"))), [("hello", "there"), ("hi",)])
        self.assertEqual(list(data.other_sents("a")), [["bye", "now"]])
        copy = data.copy()
        data.remove_lines("b")
        self.assertIn("b", copy.sent_lists)
        self.assertNotIn("b", data.sent_lists)


class TestWeightsParsedInOnePass(unittest.TestCase):
    def test_weights_equal_the_per_token_float_parse(self):
        from ovos_padatious.fann import _parse_weights
        text = "(0, 0.5) (1, -1.25e-3) (2, 3.0) (3, 1e10)"
        self.assertEqual(_parse_weights(text).tolist(), [0.5, -1.25e-3, 3.0, 1e10])
