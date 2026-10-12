"""Boot-time registration cost: what a restart with unchanged skills must not
redo.

ovos-workshop sends every intent twice, on the legacy
``padatious:register_intent`` topic and on OVOS-INTENT-4's
``ovos.intent.register.template``, with the same samples. Both used to be
expanded, normalised, hashed and loaded from the cache, and the second load
parsed every network of the intent again. On an 18-skill runtime serving 11
languages that doubled most of the plugin's share of a restart.
"""
import shutil
import tempfile
import time
import unittest
from unittest import mock

from ovos_bus_client.message import Message
from ovos_utils.messagebus import FakeBus
from xxhash import xxh32

from ovos_padatious import padaos
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


class TestDualEmittedIntentAppliedOnce(_Pipeline):
    def test_second_wire_contract_copy_does_not_reach_the_container(self):
        with mock.patch.object(IntentContainer, "add_intent",
                               autospec=True,
                               side_effect=IntentContainer.add_intent) as add:
            self.pipeline.register_intent(legacy_msg())
            self.pipeline.handle_register_template(spec_msg())
        self.assertEqual(add.call_count, 1)
        self.settle()
        match = self.pipeline.calc_intent("what time is it", lang=LANG)
        self.assertEqual(match.name, f"{SKILL_ID}:time")

    def test_metadata_still_recorded_from_every_copy(self):
        self.pipeline.register_intent(legacy_msg())
        msg = spec_msg()
        msg.data["requires_context"] = [{"key": "topic"}]
        self.pipeline.handle_register_template(msg)
        self.assertIn(f"{SKILL_ID}:time", self.pipeline._intent_context_gates)

    def test_changed_samples_are_applied(self):
        self.pipeline.register_intent(legacy_msg())
        self.settle()
        with mock.patch.object(IntentContainer, "add_intent",
                               autospec=True,
                               side_effect=IntentContainer.add_intent) as add:
            self.pipeline.handle_register_template(spec_msg(["what hour is it"]))
        self.assertEqual(add.call_count, 1)

    def test_detach_then_identical_registration_applies_again(self):
        self.pipeline.register_intent(legacy_msg())
        self.settle()
        self.pipeline.handle_detach_intent(Message(
            "detach_intent", {"intent_name": f"{SKILL_ID}:time.intent"}))
        with mock.patch.object(IntentContainer, "add_intent",
                               autospec=True,
                               side_effect=IntentContainer.add_intent) as add:
            self.pipeline.register_intent(legacy_msg())
        self.assertEqual(add.call_count, 1)

    def test_rejected_registration_does_not_shadow_a_later_one(self):
        with mock.patch.object(PadatiousPipeline, "_unpack_object", return_value=None):
            self.pipeline.register_intent(legacy_msg())
        with mock.patch.object(IntentContainer, "add_intent",
                               autospec=True,
                               side_effect=IntentContainer.add_intent) as add:
            self.pipeline.handle_register_template(spec_msg())
        self.assertEqual(add.call_count, 1)


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


class TestPrefilterBuiltOnFirstUse(unittest.TestCase):
    def test_compile_leaves_the_prefilter_for_the_first_query(self):
        c = padaos.IntentContainer()
        c.add_intent("a", ["turn on the light", "lights on"])
        c.compile()
        self.assertIs(c.intents["a"]._prefilter, padaos._IntentRegexes._UNBUILT)
        self.assertEqual([m["name"] for m in c.calc_intents("lights on")], ["a"])
        self.assertIsNotNone(c.intents["a"]._prefilter)
        self.assertIsNot(c.intents["a"]._prefilter, padaos._IntentRegexes._UNBUILT)


class TestSecondaryLanguagesWaitForTheBoot(unittest.TestCase):
    """A secondary language's padaos compile waits for ``mycroft.ready`` (or
    for the plugin to go quiet); the primary language and any network that
    needs training never wait."""

    def setUp(self):
        self.cache = tempfile.mkdtemp()
        conf = {"lang": "en-US", "secondary_langs": ["fr-FR"]}
        with mock.patch("ovos_padatious.opm.Configuration", return_value=conf):
            self.pipeline = PadatiousPipeline(FakeBus(), config={"intent_cache": self.cache})
        # the hold must outlast the test, so only `mycroft.ready` releases it
        hold = mock.patch.object(PadatiousPipeline, "_SECONDARY_QUIET_S", 60.0)
        hold.start()
        self.addCleanup(hold.stop)
        debounce = mock.patch.object(IntentContainer, "_TRAIN_DEBOUNCE_S", 0.05)
        debounce.start()
        self.addCleanup(debounce.stop)

    def tearDown(self):
        self.pipeline.shutdown()
        shutil.rmtree(self.cache, ignore_errors=True)

    def register(self, lang, name="time", samples=SAMPLES):
        self.pipeline.register_intent(Message("padatious:register_intent", {
            "skill_id": SKILL_ID, "name": f"{SKILL_ID}:{name}", "lang": lang,
            "samples": list(samples)}, {"skill_id": SKILL_ID}))

    def wait(self, condition, timeout=20.0):
        deadline = time.monotonic() + timeout
        while not condition() and time.monotonic() < deadline:
            time.sleep(0.05)
        return condition()

    def warm_restart(self):
        """Train both languages once, then start over on the same cache: the
        networks load from it, so only padaos is left to compile."""
        self.pipeline.handle_ready()
        self.register("en-US")
        self.register("fr-FR")
        self.assertTrue(self.pipeline.wait_until_trained(timeout=30))
        self.pipeline.shutdown()
        conf = {"lang": "en-US", "secondary_langs": ["fr-FR"]}
        with mock.patch("ovos_padatious.opm.Configuration", return_value=conf):
            self.pipeline = PadatiousPipeline(FakeBus(), config={"intent_cache": self.cache})
        self.register("en-US")
        self.register("fr-FR")

    def test_secondary_padaos_waits_for_ready_primary_does_not(self):
        self.warm_restart()
        en, fr = self.pipeline.containers["en-US"], self.pipeline.containers["fr-FR"]
        self.assertTrue(self.wait(lambda: not en.needs_compile))
        time.sleep(0.5)
        self.assertTrue(fr.needs_compile, "the secondary language compiled before ready")
        self.assertFalse(fr.must_train)
        # the primary's prefilters were built in the background
        self.assertIsNot(next(iter(en.padaos.intents.values()))._prefilter,
                         padaos._IntentRegexes._UNBUILT)
        self.pipeline.bus.emit(Message("mycroft.ready"))
        self.assertTrue(self.wait(lambda: not fr.needs_compile))
        self.assertTrue(self.wait(lambda: next(iter(fr.padaos.intents.values()))._prefilter
                                  is not padaos._IntentRegexes._UNBUILT))

    def test_a_network_to_train_never_waits(self):
        self.register("fr-FR", name="fresh", samples=["bonjour tout le monde"])
        fr = self.pipeline.containers["fr-FR"]
        self.assertTrue(self.wait(lambda: not fr.needs_compile))

    def test_quiet_plugin_settles_without_ready(self):
        self.warm_restart()
        fr = self.pipeline.containers["fr-FR"]
        with mock.patch.object(PadatiousPipeline, "_SECONDARY_QUIET_S", 0.3):
            self.assertTrue(self.wait(lambda: not fr.needs_compile))
        self.assertTrue(self.pipeline._boot_settled.is_set())


class TestPadaosPatternsSurviveARestart(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.file = self.dir + "/padaos-patterns.json"

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def compiled(self, **kw):
        c = padaos.IntentContainer(pattern_cache=self.file)
        c.add_intent("a", ["turn on the (light|lamp)", "lights on"])
        c.add_entity("room", ["kitchen", "hall"])
        c.add_intent("b", ["clean the {room}"])
        c.compile()
        return c

    def test_a_restart_rewrites_no_line(self):
        first = self.compiled()
        with mock.patch.object(padaos.IntentContainer, "_create_pattern",
                               side_effect=AssertionError("rewrote a cached line")):
            second = self.compiled()
        self.assertEqual({k: [r.pattern for r in v] for k, v in first.intents.items()},
                         {k: [r.pattern for r in v] for k, v in second.intents.items()})
        self.assertEqual([m["name"] for m in second.calc_intents("clean the kitchen")], ["b"])

    def test_other_rules_or_a_broken_file_are_ignored(self):
        with open(self.file, "w") as f:
            f.write('{"rules": -1, "patterns": {"lights on": "^nope$"}}')
        c = self.compiled()
        self.assertEqual([m["name"] for m in c.calc_intents("lights on")], ["a"])
        with open(self.file, "w") as f:
            f.write("not json")
        c = self.compiled()
        self.assertEqual([m["name"] for m in c.calc_intents("lights on")], ["a"])


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
