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

import json
from typing import Optional, Dict, List, Any

from ovos_padatious.util import StrEnum


class IdManager:
    """
    Manages specific unique identifiers for tokens.
    Used to convert tokens to vectors.

    Args:
        id_cls (Type[StrEnum]): The class to use for token identifiers (default: StrEnum).
        ids (Optional[Dict[str, int]]): Pre-existing dictionary of token identifiers (default: None).
    """

    def __init__(self, id_cls: type = StrEnum, ids: Optional[Dict[str, int]] = None) -> None:
        if ids is not None:
            self.ids = ids
        else:
            self.ids = {}
            for i in id_cls.values():
                self.add_token(i)

    def __len__(self) -> int:
        """Returns the number of unique tokens managed."""
        return len(self.ids)

    @staticmethod
    def adj_token(token: str) -> str:
        """
        Adjusts a token by replacing any digits with '#'.

        Args:
            token (str): The token to adjust.

        Returns:
            str: The adjusted token.
        """
        if token.isdigit():
            for i in range(10):
                if str(i) in token:
                    token = token.replace(str(i), '#')
        return token

    def vector(self) -> List[float]:
        """Creates a zeroed vector with the same length as the number of tokens."""
        return [0.0] * len(self.ids)

    def save(self, prefix: str) -> None:
        """
        Saves the token identifiers to a file.

        Args:
            prefix (str): The prefix for the file name.
        """
        with open(prefix + '.ids', 'w') as f:
            json.dump(self.ids, f)

    def load(self, prefix: str) -> None:
        """
        Loads token identifiers from a file.

        Args:
            prefix (str): The prefix for the file name.
        """
        with open(prefix + '.ids', 'r') as f:
            self.ids = json.load(f)

    def assign(self, vector: List[float], key: str, val: float) -> None:
        """
        Assigns a value to a position in the vector based on the token's identifier.

        Args:
            vector (List[float]): The vector to modify.
            key (str): The token key.
            val (float): The value to assign.
        """
        vector[self.ids[self.adj_token(key)]] = val

    def __contains__(self, token: str) -> bool:
        """Checks if a token is managed by this IdManager."""
        return self.adj_token(token) in self.ids

    def add_token(self, token: str) -> None:
        """
        Adds a new token to the manager.

        Args:
            token (str): The token to add.
        """
        token = self.adj_token(token)
        if token not in self.ids:
            self.ids[token] = len(self.ids)

    def add_sent(self, sent: List[str]) -> None:
        """
        Adds tokens from a sentence to the manager.

        Args:
            sent (List[str]): The sentence as a list of tokens.
        """
        for token in sent:
            self.add_token(token)
