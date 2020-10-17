from abc import ABC
from dataclasses import dataclass
import tensorflow as tf


class TrainBatch(ABC):
    """ The Train Batch contains all information needed
    to perform a single optimization step. """


@dataclass(frozen=True)
class ClassifierTrainBatch(TrainBatch):
    """ The Classifier Train Batch contains all information
    needed to perform a single optimization step.

    `token_ids`:
        Indices of input sequence tokens in the vocabulary.
    `attention_mask`:
        Mask to avoid performing attention on padding token indices
        (this is not related with masks from the language modeling task).
    `token_type_ids`:
        Segment token indices to indicate first and second portions of the
        inputs.
    `target_labels`:
        Target polarity labels (neutral, negative, positive).
    """
    token_ids: tf.Tensor
    attention_mask: tf.Tensor
    token_type_ids: tf.Tensor
    target_labels: tf.Tensor
