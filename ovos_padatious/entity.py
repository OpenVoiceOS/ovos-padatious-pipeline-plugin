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

from os.path import join
from typing import Any, Dict, Type

from ovos_padatious.simple_intent import SimpleIntent
from ovos_padatious.trainable import Trainable


class Entity(SimpleIntent, Trainable):
    def __init__(self, name: str, *args: Any, **kwargs: Any) -> None:
        """
        Initializes an Entity instance.

        Args:
            name (str): The name of the entity.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        SimpleIntent.__init__(self, name)
        Trainable.__init__(self, name, *args, **kwargs)

    @staticmethod
    def verify_name(token: str) -> None:
        """
        Verifies that the token is not surrounded by braces.

        Args:
            token (str): The token to verify.

        Raises:
            ValueError: If the token is surrounded by braces.
        """
        if token.startswith('{') or token.endswith('}'):
            raise ValueError('Token must not be surrounded in braces (e.g., {word} should be word)')

    @staticmethod
    def wrap_name(name: str) -> str:
        """
        Wraps the skill name and entity into a specific format.

        Args:
            name (str): The skill name or entity name.

        Returns:
            str: Wrapped name in the format SkillName:{entity}.
        """
        if ':' in name:
            parts = name.split(':')
            intent_name, ent_name = parts[0], parts[1:]
            return f"{intent_name}:{{{':'.join(ent_name)}}}"
        else:
            return f"{{{name}}}"

    def save(self, folder: str) -> None:
        """
        Saves the entity to the specified folder.

        Args:
            folder (str): The folder path where the entity should be saved.
        """
        prefix = join(folder, self.name)
        SimpleIntent.save(self, prefix)
        self.save_hash(prefix)

    @classmethod
    def from_file(cls: Type['Entity'], name: str, folder: str) -> 'Entity':
        """
        Creates an Entity instance from a file.

        Args:
            cls (Type[Entity]): The class itself.
            name (str): The name of the entity.
            folder (str): The folder path where the entity file is located.

        Returns:
            Entity: The loaded Entity instance.
        """
        self = super(Entity, cls).from_file(name, join(folder, name))
        self.load_hash(join(folder, name))
        return self
