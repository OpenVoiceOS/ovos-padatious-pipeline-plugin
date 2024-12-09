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
from functools import partial
from os.path import join, isfile, splitext
from typing import List, Type, Union

from ovos_utils.log import LOG

import ovos_padatious
from ovos_padatious.trainable import Trainable
from ovos_padatious.train_data import TrainData
from ovos_padatious.util import lines_hash


def _train_and_save(obj: Trainable, cache: str, data: TrainData, print_updates: bool) -> None:
    """
    Internal function to train objects sequentially and save them.

    Args:
        obj (Trainable): Object to train (Intent or Entity).
        cache (str): Path to the cache directory.
        data (TrainData): Training data.
        print_updates (bool): Whether to print updates during training.
    """
    obj.train(data)
    obj.save(cache)
    if print_updates:
        LOG.debug(f'Saving {obj.name} to cache ({cache})')


class TrainingManager:
    """
    Manages sequential training of either Intents or Entities.

    Args:
        cls (Type[Trainable]): Class to wrap (Intent or Entity).
        cache_dir (str): Path to the cache directory.
    """

    def __init__(self, cls: Type[Trainable], cache_dir: str) -> None:
        """
        Initializes the TrainingManager.

        Args:
            cls (Type[Trainable]): Class to be managed (Intent or Entity).
            cache_dir (str): Path where cache files are stored.
        """
        self.cls = cls
        self.cache = cache_dir
        self.objects: List[Trainable] = []
        self.objects_to_train: List[Trainable] = []
        self.train_data = TrainData()

    def add(self, name: str, lines: List[str], reload_cache: bool = False, must_train: bool = True) -> None:
        """
        Adds a new intent or entity for training or loading from cache.

        Args:
            name (str): Name of the intent or entity.
            lines (List[str]): Lines of training data.
            reload_cache (bool): Whether to force reload of cache if it exists.
            must_train (bool): Whether training is required for the new intent/entity.
        """
        if not must_train:
            LOG.debug(f"Loading {name} from intent cache")
            self.objects.append(self.cls.from_file(name=name, folder=self.cache))
        # general case: load resource (entity or intent) to training queue
        # or if no change occurred to memory data structures
        else:
            hash_fn = join(self.cache, name + '.hash')
            old_hsh = None
            if isfile(hash_fn):
                with open(hash_fn, 'rb') as g:
                    old_hsh = g.read()
            min_ver = splitext(ovos_padatious.__version__)[0]
            new_hsh = lines_hash([min_ver] + lines)
            retrain = reload_cache or old_hsh != new_hsh
            if not retrain:
                try:
                    LOG.debug(f"Loading {name} from intent cache")
                    self.objects.append(self.cls.from_file(name=name, folder=self.cache))
                except Exception as e:
                    LOG.error(f"Failed to load intent from cache: {name} - {str(e)}")
                    retrain = True
            if retrain:
                LOG.debug(f"Queuing {name} for training")
                self.objects_to_train.append(self.cls(name=name, hsh=new_hsh))
            self.train_data.add_lines(name, lines)

    def load(self, name: str, file_name: str, reload_cache: bool = False) -> None:
        """
        Loads an entity or intent from a file and adds it for training or caching.

        Args:
            name (str): Name of the intent or entity.
            file_name (str): Path to the file containing the training data.
            reload_cache (bool): Whether to reload the cache for this intent/entity.
        """
        with open(file_name) as f:
            self.add(name, f.read().split('\n'), reload_cache)

    def remove(self, name: str) -> None:
        """
        Removes an intent or entity from the training and cache.

        Args:
            name (str): Name of the intent or entity to remove.
        """
        self.objects = [i for i in self.objects if i.name != name]
        self.objects_to_train = [i for i in self.objects_to_train if i.name != name]
        self.train_data.remove_lines(name)

    def train(self, debug: bool = True, single_thread: Union[None, bool] = None,
              timeout: Union[None, int] = None) -> None:
        """
        Trains all intents and entities sequentially.

        Args:
            debug (bool): Whether to print debug messages.
            single_thread (bool): DEPRECATED
            timeout (float): DEPRECATED
        """
        if single_thread is not None:
            LOG.warning("'single_thread' argument is deprecated and will be ignored")
        if timeout is not None:
            LOG.warning("'timeout' argument is deprecated and will be ignored")

        train = partial(_train_and_save, cache=self.cache, data=self.train_data, print_updates=debug)

        # Train objects sequentially
        for obj in self.objects_to_train:
            try:
                train(obj)
            except Exception as e:
                LOG.error(f"Error training {obj.name}: {e}")

        # Load saved objects from disk
        for obj in self.objects_to_train:
            try:
                self.objects.append(self.cls.from_file(name=obj.name, folder=self.cache))
            except Exception as e:
                LOG.error(f"Failed to load trained object {obj.name}: {e}")
        self.objects_to_train = []
