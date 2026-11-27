"""padaos: prefilters are built in the background and template patterns survive a restart."""
import shutil
import tempfile
import time
import unittest
from unittest import mock

from ovos_bus_client.message import Message
from ovos_utils.messagebus import FakeBus

from ovos_padatious import padaos
from ovos_padatious.intent_container import IntentContainer
from ovos_padatious.opm import PadatiousPipeline

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


class TestPrefilterBuiltInTheBackground(unittest.TestCase):
    def test_a_query_neither_needs_nor_builds_the_prefilter(self):
        c = padaos.IntentContainer()
        c.add_intent("a", ["turn on the light", "lights on"])
        c.add_intent("b", ["turn off the light"])
        c.compile()
        self.assertIs(c.intents["a"]._prefilter, padaos._IntentRegexes._UNBUILT)
        self.assertEqual([m["name"] for m in c.calc_intents("lights on")], ["a"])
        self.assertIs(c.intents["a"]._prefilter, padaos._IntentRegexes._UNBUILT)
        # the background warm builds it; the answer stays the same
        _ = c.intents["a"].prefilter
        self.assertIsNot(c.intents["a"]._prefilter, padaos._IntentRegexes._UNBUILT)
        self.assertEqual([m["name"] for m in c.calc_intents("lights on")], ["a"])
        self.assertEqual([m["name"] for m in c.calc_intents("turn off the light")], ["b"])


class TestPrefiltersBuiltAfterCompile(unittest.TestCase):
    """Every configured language is registered and compiled the same way, as
    soon as its registrations go quiet; the plugin waits for no event."""
    def setUp(self):
        self.cache = tempfile.mkdtemp()
        debounce = mock.patch.object(IntentContainer, "_TRAIN_DEBOUNCE_S", 0.05)
        debounce.start()
        self.addCleanup(debounce.stop)
        self.pipeline = self.new_pipeline()
    def tearDown(self):
        self.pipeline.shutdown()
        shutil.rmtree(self.cache, ignore_errors=True)
    def new_pipeline(self):
        conf = {"lang": "en-US", "secondary_langs": ["fr-FR"]}
        with mock.patch("ovos_padatious.opm.Configuration", return_value=conf):
            return PadatiousPipeline(FakeBus(), config={"intent_cache": self.cache})
    def register(self, lang, name="time", samples=SAMPLES):
        self.pipeline.register_intent(Message("padatious:register_intent", {
            "skill_id": SKILL_ID, "name": f"{SKILL_ID}:{name}", "lang": lang,
            "samples": list(samples)}, {"skill_id": SKILL_ID}))
    def wait(self, condition, timeout=20.0):
        deadline = time.monotonic() + timeout
        while not condition() and time.monotonic() < deadline:
            time.sleep(0.05)
        return condition()
    def test_prefilters_are_built_in_the_background(self):
        # a line with a digit compiles to a regex (plain lines need none)
        self.register("fr-FR", name="count", samples=["count to 3", "count to 4"])
        fr = self.pipeline.containers["fr-FR"]
        self.assertTrue(self.wait(lambda: not fr.needs_compile and fr.padaos.intents))
        self.assertTrue(self.wait(lambda: next(iter(fr.padaos.intents.values()))._prefilter
                                  is not padaos._IntentRegexes._UNBUILT))


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
