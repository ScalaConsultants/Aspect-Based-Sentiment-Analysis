from dataclasses import dataclass
from typing import List
from typing import Iterable

import tensorflow as tf
import transformers

from .. import ClassifierExample
from .. import InMemoryDataset
from .. import TrainBatch


@dataclass(frozen=True)
class ClassifierTrainBatch(TrainBatch):
    """ The Classifier Train Batch contains all information needed
    to perform a single optimization step.

    `token_ids`:
        Indices of input sequence tokens in the vocabulary.
    `attention_mask`:
        Mask to avoid performing attention on padding token indices
        (this is not related with masks from the language modeling task).
    `token_type_ids`:
        Segment token indices to indicate first and second portions of the inputs.
    `target_labels`:
        Target polarity labels (neutral, negative, positive).
    """
    token_ids: tf.Tensor
    attention_mask: tf.Tensor
    token_type_ids: tf.Tensor
    target_labels: tf.Tensor


@dataclass(frozen=True)
class ClassifierDataset(InMemoryDataset):
    examples: List[ClassifierExample]
    batch_size: int
    tokenizer: transformers.PreTrainedTokenizer
    num_polarities: int = 3

    def preprocess_batch(
            self, batch_examples: List[ClassifierExample]
    ) -> ClassifierTrainBatch:
        """ Convert classifier model examples to the ClassifierTrainBatch. """
        pairs = [(e.text, e.aspect) for e in batch_examples]
        encoded = self.tokenizer.batch_encode_plus(pairs,
                                                   add_special_tokens=True,
                                                   pad_to_max_length=True,
                                                   return_attention_masks=True,
                                                   return_tensors='tf')
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        token_type_ids = encoded['token_type_ids']
        target_labels = tf.one_hot([e.sentiment for e in batch_examples],
                                   depth=self.num_polarities)

        return ClassifierTrainBatch(
            input_ids,
            attention_mask,
            token_type_ids,
            target_labels
        )

    @classmethod
    def from_iterable(cls, examples: Iterable[ClassifierExample], *args, **kwargs):
        """ For simplification, we materialize the iterable of examples to have
        straightforward control over the processing examples order """
        examples = list(examples)
        return cls(examples, *args, **kwargs)
