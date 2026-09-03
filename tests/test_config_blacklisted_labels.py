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
