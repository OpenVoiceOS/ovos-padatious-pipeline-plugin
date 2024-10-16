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

from abc import ABC, abstractmethod
from typing import Any


class Trainable(ABC):
    """
    Abstract base class for trainable models. Implements basic hash handling and
    requires subclasses to implement training, saving, and loading methods.
    """

    def __init__(self, name: str, hsh: bytes = b''):
        """
        Initialize a Trainable instance.

        Args:
            name (str): Name of the model.
            hsh (bytes): Optional hash value for the model (defaults to an empty byte string).
        """
        self.name = name
        self.hash = hsh

    def load_hash(self, prefix: str) -> None:
        """
        Load the model's hash from a file.

        Args:
            prefix (str): Path prefix where the hash file is located.
        """
        with open(f"{prefix}.hash", 'rb') as f:
            self.hash = f.read()

    def save_hash(self, prefix: str) -> None:
        """
        Save the model's hash to a file.

        Args:
            prefix (str): Path prefix where the hash file will be saved.
        """
        with open(f"{prefix}.hash", 'wb') as f:
            f.write(self.hash)

    @abstractmethod
    def train(self, data: Any) -> None:
        """
        Train the model on the provided data.

        Args:
            data (Any): The training data for the model.
        """
        pass

    @abstractmethod
    def save(self, prefix: str) -> None:
        """
        Save the trained model to a file.

        Args:
            prefix (str): Path prefix where the model should be saved.
        """
        pass

    @classmethod
    @abstractmethod
    def from_file(cls, name: str, folder: str) -> "Trainable":
        """
        Load a model from a file.

        Args:
            name (str): Name of the model.
            folder (str): Folder where the model is stored.

        Returns:
            Trainable: An instance of the loaded model.
        """
        pass
