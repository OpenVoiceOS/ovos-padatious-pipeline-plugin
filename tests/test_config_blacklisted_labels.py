# Copyright 2020 Mycroft AI Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Config-level ``blacklisted_labels`` (intents.ovos-padatious-pipeline-plugin).

Enables the deployment pattern where an m2v classifier fronts the default
skills' trained label set and padatious only trains/matches the labels it
still owns (e.g. user-installed skills). An entry may be an exact
``<skill_id>:<name>`` id or an fnmatch glob (``<skill_id>:*`` blacklists a
whole skill). A blacklisted label must never train, and a registration for
it must be silently ignored (debug-logged); a label blacklisted after a
container already trained it must also never be returned at match time.
"""
from unittest import TestCase, mock

from ovos_bus_client.message import Message

import ovos_padatious.opm as opm
from ovos_padatious.opm import PadatiousPipeline

LANG = "en-US"


def register_msg(skill_id, name, samples):
    full = f"{skill_id}:{name}"
    data = {"skill_id": skill_id, "name": full, "lang": LANG, "samples": samples}
    return Message("padatious:register_intent", data, {"skill_id": skill_id})


class TestConfigBlacklistedLabels(TestCase):
    def _pipeline(self, blacklisted_labels):
        config = {"blacklisted_labels": blacklisted_labels}
        return PadatiousPipeline(mock.Mock(), config=config)

    def test_exact_label_never_trains_or_matches(self):
        """A label listed exactly is not registered/trained; the other intent
        still trains and matches normally."""
        pipeline = self._pipeline(["skillA.openvoiceos:blocked"])
        pipeline.register_intent(register_msg("skillA.openvoiceos", "blocked",
                                               ["this is blocked", "another blocked test"]))
        pipeline.register_intent(register_msg("skillA.openvoiceos", "allowed",
                                               ["this is allowed", "another allowed test"]))
        container = pipeline.containers[LANG]

        self.assertNotIn("skillA.openvoiceos:blocked", pipeline.registered_intents)
        self.assertIn("skillA.openvoiceos:allowed", pipeline.registered_intents)

        container.train(single_thread=True, timeout=120)
        matches = container.calc_intents("this is blocked")
        self.assertEqual([m for m in matches if m.name == "skillA.openvoiceos:blocked"], [])
        self.assertEqual(container.calc_intent("this is allowed").name,
                         "skillA.openvoiceos:allowed")

    def test_glob_pattern_blacklists_whole_skill(self):
        """A glob pattern blacklists every intent of one skill while another
        skill's intents remain fully trainable/matchable."""
        pipeline = self._pipeline(["ovos-skill-weather.openvoiceos:*"])
        pipeline.register_intent(register_msg("ovos-skill-weather.openvoiceos", "forecast",
                                               ["what is the weather", "will it rain today"]))
        pipeline.register_intent(register_msg("ovos-skill-weather.openvoiceos", "temperature",
                                               ["what is the temperature", "how hot is it"]))
        pipeline.register_intent(register_msg("ovos-skill-news.openvoiceos", "headlines",
                                               ["tell me the news", "what are the headlines"]))
        container = pipeline.containers[LANG]

        self.assertFalse(any(n.startswith("ovos-skill-weather.openvoiceos:")
                             for n in pipeline.registered_intents))
        self.assertIn("ovos-skill-news.openvoiceos:headlines", pipeline.registered_intents)

        container.train(single_thread=True, timeout=120)
        weather_matches = [m for m in container.calc_intents("what is the weather")
                          if m.name.startswith("ovos-skill-weather.openvoiceos:")]
        temp_matches = [m for m in container.calc_intents("what is the temperature")
                        if m.name.startswith("ovos-skill-weather.openvoiceos:")]
        self.assertEqual(weather_matches, [])
        self.assertEqual(temp_matches, [])
        self.assertEqual(container.calc_intent("tell me the news").name,
                         "ovos-skill-news.openvoiceos:headlines")

    def test_matches_the_dealiased_intent_suffix_form(self):
        """The blacklist config is written the natural way a user would copy
        it (bare ``skill:name``, no ``.intent`` suffix). Registration must
        collapse the legacy ``.intent``-suffixed wire name onto that same
        canonical form BEFORE the blacklist check runs, or the entry would
        silently never match (OVOS-INTENT-1 alias-collapse quirk)."""
        pipeline = self._pipeline(["skillB.openvoiceos:greet"])
        # emitted the way ovos-workshop's legacy padatious contract does,
        # with the '.intent' suffix still attached
        msg = register_msg("skillB.openvoiceos", "greet.intent",
                           ["hello there", "hi there"])
        pipeline.register_intent(msg)

        self.assertNotIn("skillB.openvoiceos:greet", pipeline.registered_intents)
        self.assertFalse(any(n.startswith("skillB.openvoiceos:")
                             for n in pipeline.registered_intents))

    def test_intent_suffixed_config_entry_blacklists_canonical_registration(self):
        """A config entry copied with the legacy '.intent' suffix
        (e.g. 'gate.skill:turn_on.intent') must still blacklist the
        canonical 'gate.skill:turn_on' registration - a denylist must not
        fail open just because it was written with the stale suffix."""
        pipeline = self._pipeline(["gate.skill:turn_on.intent"])
        pipeline.register_intent(register_msg("gate.skill", "turn_on",
                                               ["turn it on", "switch it on"]))
        self.assertNotIn("gate.skill:turn_on", pipeline.registered_intents)

    def test_intent_suffixed_config_entry_blacklists_intent_suffixed_wire_registration(self):
        """The same suffixed config entry also blacklists a registration
        that itself still arrives with the legacy '.intent' suffix."""
        pipeline = self._pipeline(["gate.skill:turn_on.intent"])
        msg = register_msg("gate.skill", "turn_on.intent",
                            ["turn it on", "switch it on"])
        pipeline.register_intent(msg)
        self.assertFalse(any(n.startswith("gate.skill:")
                             for n in pipeline.registered_intents))

    def test_glob_with_intent_suffix_behaves_like_bare_glob(self):
        """'gate.skill:*.intent' behaves exactly like 'gate.skill:*': the
        trailing '.intent' is stripped from the pattern the same way it is
        from an exact id, since dealiasing is a pure suffix strip."""
        pipeline = self._pipeline(["gate.skill:*.intent"])
        pipeline.register_intent(register_msg("gate.skill", "turn_on",
                                               ["turn it on", "switch it on"]))
        pipeline.register_intent(register_msg("gate.skill", "turn_off",
                                               ["turn it off", "switch it off"]))
        self.assertFalse(any(n.startswith("gate.skill:")
                             for n in pipeline.registered_intents))

    def test_intent_suffixed_config_entry_warns_once(self):
        """Canonicalizing a suffixed blacklisted_labels entry logs a single
        deprecation warning naming the offending entry and its canonical
        replacement."""
        opm._warned_legacy_blacklist_entries.discard("gate.skill:warn_me.intent")
        with mock.patch.object(opm.LOG, "warning") as warn:
            self._pipeline(["gate.skill:warn_me.intent"])
            self._pipeline(["gate.skill:warn_me.intent"])
        matching = [c for c in warn.call_args_list
                    if "gate.skill:warn_me.intent" in c.args[0]]
        self.assertEqual(len(matching), 1)
        self.assertIn("gate.skill:warn_me", matching[0].args[0])

    def test_match_emission_falls_through_to_next_candidate(self):
        """Defense in depth: a label blacklisted after its container already
        trained it (e.g. a stale intent_cache from before the config entry
        was added) must never be returned by calc_intent, which must fall
        through to the next surviving candidate rather than the whole query
        going stale. This exercises the post-cache filtering in calc_intent,
        not just the registration-time refusal to train."""
        pipeline = self._pipeline([])
        pipeline.register_intent(register_msg("skillD.openvoiceos", "blocked",
                                               ["do the blocked thing"]))
        pipeline.register_intent(register_msg("skillD.openvoiceos", "allowed",
                                               ["do the allowed thing"]))
        container = pipeline.containers[LANG]
        container.train(single_thread=True, timeout=120)

        # simulate the label being blacklisted AFTER the container already
        # trained it, without touching the module-global _calc_padatious_intent
        # lru_cache at all
        pipeline._label_blacklist = ("skillD.openvoiceos:blocked",)

        # calc_intent returns the best candidate at ANY confidence (the
        # conf_* thresholds are applied by match_high/medium/low, not here),
        # so the assertion is that the blacklisted label is never returned -
        # not that there is no match at all.
        result = pipeline.calc_intent("do the blocked thing")
        self.assertNotEqual(getattr(result, "name", None), "skillD.openvoiceos:blocked")
        self.assertEqual(pipeline.calc_intent("do the allowed thing").name,
                         "skillD.openvoiceos:allowed")

    def test_no_config_behaves_like_today(self):
        """Empty/absent 'blacklisted_labels' changes nothing: registration
        and matching behave exactly as without the feature."""
        pipeline = PadatiousPipeline(mock.Mock())
        pipeline.register_intent(register_msg("skillC.openvoiceos", "hello",
                                               ["hello world", "hi world"]))
        container = pipeline.containers[LANG]
        self.assertIn("skillC.openvoiceos:hello", pipeline.registered_intents)
        container.train(single_thread=True, timeout=120)
        self.assertEqual(container.calc_intent("hello world").name,
                         "skillC.openvoiceos:hello")
