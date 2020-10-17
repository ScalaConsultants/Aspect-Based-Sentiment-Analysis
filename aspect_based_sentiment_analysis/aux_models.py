from abc import ABC
from typing import List

import tensorflow as tf
from transformers import TFPreTrainedModel

from .data_types import Pattern
from .data_types import TokenizedExample


def get_key_set(patterns: List[Pattern], n: int, k: int = 1):
    """

    Parameters
    ----------
    patterns
    n
        The number of elements in the key set.
    k
        The number of the sorted (from the most important) candidates
        of the key sets.

    Returns
    -------

    """


class ReferenceRecognizer(ABC):
    """ """

    def __call__(
            self,
            example: TokenizedExample,
            hidden_states: tf.Tensor,
            attentions: tf.Tensor,
            attention_grads: tf.Tensor
    ) -> bool:
        """ """
        if not aspect:
            return True


class PatternRecognizer(ABC):
    """ """

    def __call__(
            self,
            example: TokenizedExample,
            hidden_states: tf.Tensor,
            attentions: tf.Tensor,
            attention_grads: tf.Tensor
    ) -> List[Pattern]:
        """ """


class BasicReferenceRecognizer(ReferenceRecognizer, TFPreTrainedModel):
    """
    Briefly, it represents a text and an aspect as two vectors, and predicts
    that a text relates to an aspect if the cosine similarity is bigger than
    a threshold. It calculates text and aspect representations by summing
    their subtoken vectors, context-independent embeddings that come from the
    embedding first layer.

    This model has only one parameter, nonetheless, we show how to take a use
    of the methods `save_pretrained` and `load_pretrained`. They are useful
    especially for more complex models.
    """
