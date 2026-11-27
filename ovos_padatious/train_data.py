# Copyright 2017 Mycroft AI, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ovos_utils.log import LOG

from ovos_padatious.util import tokenize, expand_lines, warn_hash_wildcard


class TrainData:
    """
    Training data used to access collections
    of tokenized sentences in intent files

    Lines are tokenized the first time something reads them. Only training
    reads them, and a restart with unchanged skills loads every model from
    the cache: tokenizing every registration's lines up front was more than
    a tenth of the plugin's share of such a restart.
    """

    def __init__(self):
        self._sents = {}
        #: name -> raw lines not expanded and tokenized yet
        self._pending = {}

    @property
    def sent_lists(self):
        for name in list(self._pending):
            self._materialize(name)
        return self._sents

    @sent_lists.setter
    def sent_lists(self, value):
        self._sents = value
        self._pending = {}

    def copy(self) -> 'TrainData':
        # NOTE: new samples might be added to TrainData object during obj.train(data)
        # which would result in ovos_padatious.training_manager:train:153 - ERROR - Error training XXX: dictionary changed size during iteration
        data = TrainData()
        data._sents = dict(self._sents)
        data._pending = dict(self._pending)
        return data

    def add_lines(self, name, lines):
        warn_hash_wildcard(name, lines)
        lines = [line for line in expand_lines(lines) if line.strip()]
        if not lines:
            # every line was malformed and skipped by expand_lines: adding
            # an empty entry would register a dead intent/entity that can
            # never match (conf 0.0 forever) instead of surfacing the
            # problem, so refuse it outright and drop any stale entry.
            # OVOS-INTENT-4 §5.3: the rejecting plugin logs at WARN.
            LOG.warning(
                "%r has no valid samples after skipping malformed template "
                "lines - not registering", name,
            )
            self.remove_lines(name)
            return
        self._sents.pop(name, None)
        self._pending[name] = lines

    def _materialize(self, name):
        lines = self._pending.pop(name, None)
        if lines is not None:
            self._sents[name] = [tokenize(line) for line in lines]

    def remove_lines(self, name):
        self._pending.pop(name, None)
        self._sents.pop(name, None)

    def add_file(self, name, file_name):
        with open(file_name, 'r') as f:
            self.add_lines(name, f.readlines())

    def all_sents(self):
        for _, sents in self.sent_lists.items():
            for i in sents:
                yield i

    def my_sents(self, my_name):
        self._materialize(my_name)
        for i in self._sents[my_name]:
            yield i

    def other_sents(self, my_name):
        for name, sents in self.sent_lists.items():
            if name != my_name:
                for i in sents:
                    yield i
