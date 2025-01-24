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
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List
from ovos_padatious.intent import Intent
from ovos_padatious.match_data import MatchData
from ovos_padatious.training_manager import TrainingManager
from ovos_padatious.util import tokenize
from ovos_utils.log import LOG


class IntentManager(TrainingManager):
    """
    Manages intents and performs matching using the OVOS Padatious framework.

    Args:
        cache (str): Path to the cache directory for storing trained models.
    """
    def __init__(self, cache: str, debug: bool = False):
        super().__init__(Intent, cache)
        self.debug = debug

    @property
    def intent_names(self):
        return [i.name for i in self.objects + self.objects_to_train]

    def calc_intents(self, query: str, entity_manager) -> List[MatchData]:
        """
        Calculate matches for the given query against all registered intents.

        Args:
            query (str): The input query to match.
            entity_manager: The entity manager for resolving entities in the query.

        Returns:
            List[MatchData]: A list of matches sorted by confidence.
        """
        sent = tokenize(query)

        def match_intent(intent):
            start_time = time.monotonic()
            try:
                match = intent.match(sent, entity_manager)
                match.detokenize()
                if self.debug:
                    LOG.debug(f"Inference for intent '{intent.name}' took {time.monotonic() - start_time} seconds")
                return match
            except Exception as e:
                LOG.error(f"Error processing intent '{intent.name}': {e}")
                return None

        # Parallelize matching
        with ThreadPoolExecutor() as executor:
            matches = list(executor.map(match_intent, self.objects))

        # Filter out None results from failed matches
        matches = [match for match in matches if match]

        return matches
