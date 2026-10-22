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
"""The inline '#' digit wildcard in template lines is a padatious-only
extension: `ovos_padatious.padaos` compiles it to a digit-class regex and
`id_manager`/`util` canonicalize literal digits to '#' for the neural net.
No other OVOS intent engine understands the syntax, it collides with the
'#'-as-comment-marker convention used elsewhere, and it assumes the ASR
output is a literal digit string. It is deprecated for one stable cycle in
favour of a `{slot}` placeholder with skill-side number parsing; matching
behavior is unchanged this cycle.
"""
import shutil
import tempfile
from unittest import TestCase, mock

from ovos_padatious.intent_container import IntentContainer

NAME = "count"


class TestHashWildcardDeprecation(TestCase):
    def setUp(self):
        self.cache_dir = tempfile.mkdtemp()
        self.cont = IntentContainer(self.cache_dir)

    def tearDown(self):
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_inline_hash_still_matches_digits_and_warns_once(self):
        with mock.patch("ovos_padatious.util.log_deprecation") as fake_warn:
            self.cont.add_intent(NAME, ["count to #"])
        fake_warn.assert_called_once()
        logged = " ".join(str(a) for a in fake_warn.call_args[0])
        self.assertIn(NAME, logged)
        self.assertIn("count to #", logged)

        self.cont.train(False)
        match = self.cont.calc_intent("count to 5")
        self.assertEqual(match.name, NAME)
        self.assertGreater(match.conf, 0)

    def test_leading_comment_marker_does_not_warn(self):
        with mock.patch("ovos_padatious.util.log_deprecation") as fake_warn:
            self.cont.add_intent("comment", ["// count to #", "# count to #", "hello there"])
        fake_warn.assert_not_called()

    def test_escaped_hash_does_not_warn(self):
        with mock.patch("ovos_padatious.util.log_deprecation") as fake_warn:
            self.cont.add_intent("escaped", [r"count to \#"])
        fake_warn.assert_not_called()

    def test_no_hash_does_not_warn(self):
        with mock.patch("ovos_padatious.util.log_deprecation") as fake_warn:
            self.cont.add_intent("plain", ["hello there"])
        fake_warn.assert_not_called()
