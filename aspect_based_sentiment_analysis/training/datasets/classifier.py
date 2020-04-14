from dataclasses import dataclass
from typing import List
from typing import Iterable

import tensorflow as tf
import transformers

from ...data_types import LabeledExample
from ..data_types import ClassifierTrainBatch
from .datasets import InMemoryDataset


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
            pad_to_max_length=True,
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
