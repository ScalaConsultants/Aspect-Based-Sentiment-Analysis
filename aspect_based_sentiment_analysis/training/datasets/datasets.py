from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Iterable
from typing import Iterator
from typing import List

import numpy as np

from ..data_types import TrainBatch


class Dataset(ABC):

    @abstractmethod
    def __iter__(self) -> Iterable[TrainBatch]:
        """ We use datasets in our custom routines, where is a simple for loop,
        therefore exclusively an iterable object is required. """

    @abstractmethod
    def preprocess_batch(self, batch_examples: List[Any]) -> TrainBatch:
        """ Transform human understandable example into model
        understandable tensors. """


class InMemoryDataset(Dataset, ABC):
    examples: List
    batch_size: int

    def __iter__(self) -> Iterator[TrainBatch]:
        """ The method shuffles example for next epoch, and returns the full
        batches in each iteration. """
        order = np.random.permutation(len(self.examples))
        batch_examples = []
        for index in order:
            example = self.examples[index]
            batch_examples.append(example)
            if len(batch_examples) == self.batch_size:
                batch = self.preprocess_batch(batch_examples)
                yield batch
                batch_examples = []


class StreamDataset(Dataset, ABC):
    batch_size: int

    def __iter__(self) -> Iterator[TrainBatch]:
        """ Produce train batches on the fly. """
        examples = self.examples_generator()
        batch_examples = []
        for example in examples:
            batch_examples.append(example)
            if len(batch_examples) == self.batch_size:
                batch_input = self.preprocess_batch(batch_examples)
                yield batch_input
                batch_examples = []

    @abstractmethod
    def examples_generator(self) -> Iterable[Any]:
        """ Stream example from a data source. """
