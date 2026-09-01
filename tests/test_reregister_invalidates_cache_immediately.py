"""A re-registration of an existing intent/entity name with NEW samples must
retire the OLD template's answer immediately, at the pipeline level - not
only once the next background compile lands.

``_calc_padatious_intent``'s lru_cache is keyed on the container's
``compiled_generation`` counter, which only bumps on a completed compile.
Every removal path (``handle_detach_intent``, ``handle_deregister_*_spec``,
``handle_disable_intent_spec``) already calls
``_calc_padatious_intent.cache_clear()`` symmetrically; ``register_intent``/
``register_entity`` (and everything that funnels through them - the spec
template-registration handlers, the enable handler) did not, so a query
answered before the re-registration kept matching the retired regex at
conf 1.0 until a compile happened to land.
"""
import shutil
import tempfile
import unittest

from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus

from ovos_padatious.opm import PadatiousPipeline

SKILL_ID = "reregister.skill"


class TestReregisterInvalidatesCacheImmediately(unittest.TestCase):
    def setUp(self):
        self.cache_dir = tempfile.mkdtemp()
        self.pipeline = PadatiousPipeline(
            FakeBus(), config={"intent_cache": self.cache_dir})
        self.lang = self.pipeline.lang

    def tearDown(self):
        self.pipeline.shutdown()
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def _register(self, samples):
        self.pipeline.register_intent(Message("padatious:register_intent", {
            "name": f"{SKILL_ID}:light", "samples": samples,
            "lang": self.lang, "skill_id": SKILL_ID,
        }))

    def test_stale_conf_1_answer_does_not_survive_reregistration(self):
        self._register(["turn on the {thing}"])
        self.assertTrue(self.pipeline.wait_until_trained(timeout=15.0))

        cached = self.pipeline.calc_intent(["turn on the lamp"], self.lang)
        self.assertIsNotNone(cached)
        self.assertEqual(cached.conf, 1.0)

        # replace the intent's samples entirely - the old template
        # ("turn on the {thing}") no longer matches this name's definition
        self._register(["switch on the {thing}"])

        # queried again with the SAME utterance immediately, before any
        # compile from this second registration has landed: must NOT
        # replay the retired regex's conf-1.0 answer from the lru_cache
        immediate = self.pipeline.calc_intent(["turn on the lamp"], self.lang)
        self.assertFalse(
            immediate is not None and immediate.conf == 1.0,
            "the lru_cache kept serving the retired intent's conf-1.0 "
            "match after re-registration")


if __name__ == '__main__':
    unittest.main()
