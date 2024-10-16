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

from ovos_padatious.intent import Intent
from ovos_padatious.match_data import MatchData
from ovos_padatious.training_manager import TrainingManager
from ovos_padatious.util import tokenize
from typing import List, Optional, Any


class IntentManager(TrainingManager):
    def __init__(self, cache: Optional[dict] = None):
        """
        Initializes the IntentManager with an optional cache.

        Args:
            cache (Optional[dict]): A cache to store previously computed intents.
        """
        super().__init__(Intent, cache)

    def calc_intents(self, query: str, entity_manager: Any) -> List[MatchData]:
        """
        Calculates the matching intents for a given query.

        Args:
            query (str): The input query string to match against intents.
            entity_manager (Any): The entity manager to assist in resolving entities.

        Returns:
            List[MatchData]: A list of MatchData objects representing matched intents.
        """
        sent = tokenize(query)
        matches: List[MatchData] = []

        for intent in self.objects:
            match = intent.match(sent, entity_manager)
            match.detokenize()  # Detokenize to convert back to natural language
            matches.append(match)

        return matches
