from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Tuple
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from transformers import TFPreTrainedModel

from .data_types import Pattern
from .data_types import TokenizedExample
from .data_types import Output
from . import alignment


class ReferenceRecognizer(ABC):

    @abstractmethod
    def __call__(
            self,
            example: TokenizedExample,
            output: Output
    ) -> bool:
        pass


class PatternRecognizer(ABC):
    """ The aim of the Pattern Recognizer is to discover patterns
    that explain a model prediction. """

    @abstractmethod
    def __call__(
            self,
            example: TokenizedExample,
            output: Output
    ) -> List[Pattern]:
        """ To recognize patterns, we provide detailed information about a
        prediction, including hidden states after each layer, attentions from
        each head in each layer, and attention gradients with respect to the
        model output. The Recognizer returns the most significant patterns. """


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


@dataclass
class BasicPatternRecognizer(PatternRecognizer):
    """
    The Attention Gradient Product uses attentions and their gradients to
    discover patterns which a model uses to make a prediction. The key idea
    is to use attentions and scale them by their gradients with respect to
    the model output (attention-gradient product). The language model
    constructs an enormous amount of various relations between words.
    However, only some of them are crucial. Thanks to gradients, we can
    filter unnecessary patterns out.

    Note that this heuristic is a rough approximation. Concerns stated in
    papers like "attentions is not explainable" are still valid. To be more
    robust, we additionally use gradients and take the mean over model layers
    and heads. Moreover, we provide an exhaustive analysis how accurate this
    pattern recognizer is. Check out details on the package website.
    """
    max_patterns: int = 10
    is_pattern_scaled: bool = True
    is_pattern_rounded: bool = True
    round_decimals: int = 2

    def __call__(
            self,
            example: TokenizedExample,
            output: Output
    ) -> List[Pattern]:
        text_mask = self.text_tokens_mask(example)
        w, pattern_vectors = self.transform(output, text_mask,
                                            example.alignment)
        patterns = self.build_patterns(w, pattern_vectors)
        return patterns

    def transform(
            self,
            output: Output,
            text_mask: List[bool],
            token_subtoken_alignment: List[List[int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = output.attentions * tf.abs(output.attention_grads)
        x = tf.reduce_sum(x, axis=[0, 1], keepdims=True)
        x = alignment.merge_tensor(x, alignment=token_subtoken_alignment)
        x = x.numpy().squeeze(axis=(0, 1))

        w = x[0, text_mask]
        w /= np.max(w + 1e-9)

        patterns = x[text_mask, :][:, text_mask]
        max_values = np.max(patterns + 1e-9, axis=1)
        np.fill_diagonal(patterns, max_values)
        patterns /= max_values.reshape(-1, 1)

        if self.is_pattern_scaled:
            patterns *= w.reshape(-1, 1)
        if self.is_pattern_rounded:
            patterns = np.round(patterns, decimals=self.round_decimals)
        return w, patterns

    @staticmethod
    def text_tokens_mask(example: TokenizedExample) -> List[bool]:
        """ Get the mask of text tokens according to the BERT input. """
        mask = np.zeros(len(example.tokens)).astype(bool)
        mask[1:len(example.text_tokens) + 1] = True
        return mask.tolist()

    def build_patterns(self, w: np.ndarray, pattern_vectors: np.ndarray) -> \
    List[int]:
        pass


def predict_key_set(patterns: List[Pattern], n: int, k: int = 1):
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
