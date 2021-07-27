from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any
from typing import Iterable
from typing import Iterator
from typing import List

import numpy as np
import tensorflow as tf
import transformers

from . import ClassifierTrainBatch
from .data_types import TrainBatch
from ..data_types import LabeledExample


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


@dataclass(frozen=True)
class ClassifierDataset(InMemoryDataset):
    examples: List[LabeledExample]
    batch_size: int
    tokenizer: transformers.PreTrainedTokenizer
    num_polarities: int = 3

    def preprocess_batch(
            self, batch_examples: List[LabeledExample]
    ) -> ClassifierTrainBatch:
        """ Convert classifier model example to the ClassifierTrainBatch. """
        pairs = [(e.text, e.aspect) for e in batch_examples]
        encoded = self.tokenizer.batch_encode_plus(
            pairs,
            add_special_tokens=True,
            padding=True,
            return_attention_masks=True,
            return_tensors='tf'
        )
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        token_type_ids = encoded['token_type_ids']
        sentiments = [e.sentiment for e in batch_examples]
        target_labels = tf.one_hot(sentiments, depth=self.num_polarities)
        train_batch = ClassifierTrainBatch(
            input_ids,
            attention_mask,
            token_type_ids,
            target_labels
        )
        return train_batch

    @classmethod
    def from_iterable(cls, examples: Iterable[LabeledExample], *args, **kwargs):
        """ For simplification, we materialize the iterable of example to have
        straightforward control over the processing example order """
        examples = list(examples)
        return cls(examples, *args, **kwargs)
