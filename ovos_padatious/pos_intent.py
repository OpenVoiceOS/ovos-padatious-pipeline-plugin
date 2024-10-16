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

import math
from typing import List, Tuple, Dict, Optional, Any

from ovos_padatious.entity_edge import EntityEdge
from ovos_padatious.match_data import MatchData


class PosIntent:
    """
    A class for handling positional intents used to extract entities from sentences.

    Args:
        token (str): The token to attach to (something like {word}).
        intent_name (str): Optional name of the intent. Defaults to an empty string.
    """

    def __init__(self, token: str, intent_name: str = '') -> None:
        self.token = token
        self.edges: List[EntityEdge] = [
            EntityEdge(-1, token, intent_name),
            EntityEdge(+1, token, intent_name)
        ]

    def match(self, orig_data: Any, entity: Optional[Any] = None) -> List[MatchData]:
        """
        Matches the original data against the token and extracts entities.

        Args:
            orig_data (Any): Original data containing the sentence to match.
            entity (Optional[Any]): An entity to match against. Defaults to None.

        Returns:
            List[MatchData]: A list of possible matches with their corresponding data.
        """
        l_matches = [(self.edges[0].match(orig_data.sent, pos), pos)
                     for pos in range(len(orig_data.sent))]
        r_matches = [(self.edges[1].match(orig_data.sent, pos), pos)
                     for pos in range(len(orig_data.sent))]

        def is_valid(l_pos: int, r_pos: int) -> bool:
            """Check if the positions are valid for matching."""
            if r_pos < l_pos:
                return False
            return all(not orig_data.sent[p].startswith('{') for p in range(l_pos, r_pos + 1))

        possible_matches: List[MatchData] = []
        for l_conf, l_pos in l_matches:
            if l_conf < 0.2:
                continue
            for r_conf, r_pos in r_matches:
                if r_conf < 0.2 or not is_valid(l_pos, r_pos):
                    continue

                extracted = orig_data.sent[l_pos:r_pos + 1]

                pos_conf = (l_conf - 0.5 + r_conf - 0.5) / 2 + 0.5
                ent_conf = entity.match(extracted) if entity else 1

                new_sent = orig_data.sent[:l_pos] + [self.token] + orig_data.sent[r_pos + 1:]
                new_matches = orig_data.matches.copy()
                new_matches[self.token] = extracted

                extra_conf = math.sqrt(pos_conf * ent_conf) - 0.5
                data = MatchData(orig_data.name, new_sent, new_matches,
                                 orig_data.conf + extra_conf)
                possible_matches.append(data)

        return possible_matches

    def save(self, prefix: str) -> None:
        """
        Saves the positional intent's data.

        Args:
            prefix (str): The prefix to use for the saved data.
        """
        prefix += '.' + self.token
        for edge in self.edges:
            edge.save(prefix)

    @classmethod
    def from_file(cls, prefix: str, token: str) -> 'PosIntent':
        """
        Creates a PosIntent instance from saved data.

        Args:
            prefix (str): The prefix used for saved data.
            token (str): The token associated with the intent.

        Returns:
            PosIntent: A new instance of PosIntent.
        """
        prefix += '.' + token
        instance = cls(token)
        for edge in instance.edges:
            edge.load(prefix)
        return instance

    def train(self, train_data: Any) -> None:
        """
        Trains the positional intent on the provided training data.

        Args:
            train_data (Any): The data to train on.
        """
        for edge in self.edges:
            edge.train(train_data)
