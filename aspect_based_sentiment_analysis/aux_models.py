from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Set
from typing import Tuple
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from transformers import PretrainedConfig

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
    """
    The aim of the Pattern Recognizer is to discover patterns that explain
    a model prediction.
    """

    @abstractmethod
    def __call__(
            self,
            example: TokenizedExample,
            output: Output
    ) -> List[Pattern]:
        """
        To recognize patterns, we provide detailed information about a
        prediction, including hidden states after each layer, attentions from
        each head in each layer, and attention gradients with respect to the
        model output. The Recognizer returns the most significant patterns.
        """


@dataclass
class BasicReferenceRecognizer(ReferenceRecognizer, PretrainedConfig):
    """
    The Basic Reference Recognizer predicts whether a text relates to an
    aspect or not. Briefly, it represents a text and an aspect as two
    vectors, measure cosine similarity between them, and then use the simple
    logistic regression to make a prediction. It calculates text and aspect
    representations by summing their subtoken vectors, context-independent
    embeddings that come from the embedding first layer. This model has two
    parameter (β_0, β_1). Benefit from two useful methods `save_pretrained`
    and `load_pretrained` (to persist the model for future use).
    """
    weights: Tuple[float, float]
    model_type: str = 'reference_recognizer'

    def __call__(
            self,
            example: TokenizedExample,
            output: Output
    ) -> bool:
        β_0, β_1 = self.weights
        n = len(example.subtokens)
        hidden_states = output.hidden_states[:, :n, :]  # Trim padded tokens.
        text_mask, aspect_mask = self.text_aspect_subtoken_masks(example)
        similarity = self.transform(hidden_states, text_mask, aspect_mask)
        is_reference = β_0 + β_1 * similarity > 0
        return bool(is_reference)   # Do not use the numpy bool object.

    @staticmethod
    def transform(
            hidden_states: tf.Tensor,
            text_mask: List[bool],
            aspect_mask: List[bool]
    ) -> float:
        hidden_states = hidden_states.numpy()
        h = hidden_states[0, ...]  # Take embeddings without context.
        h_t = h[text_mask, :].mean(axis=0)
        h_a = h[aspect_mask, :].mean(axis=0)

        h_t /= np.linalg.norm(h_t, ord=2)
        h_a /= np.linalg.norm(h_a, ord=2)

        similarity = h_t @ h_a
        return similarity

    @staticmethod
    def text_aspect_subtoken_masks(
            example: TokenizedExample
    ) -> Tuple[List[bool], List[bool]]:
        text = np.zeros(len(example.subtokens)).astype(bool)
        text[1:len(example.text_subtokens)+1] = True
        aspect = np.zeros(len(example.subtokens)).astype(bool)
        aspect[-(len(example.aspect_subtokens) + 1):-1] = True
        return text.tolist(), aspect.tolist()


@dataclass
class BasicPatternRecognizer(PatternRecognizer):
    """
    The Base Pattern Recognizer uses attentions and their gradients to
    discover patterns which a model uses to make a prediction. The key idea
    is to use attentions and scale them by their gradients with respect to
    the model output (attention-gradient product). The language model
    constructs various relations between words. However, only some of them
    are crucial. Thanks to gradients, we can filter unnecessary patterns out.

    Note that this is heuristic, an approximation. Concerns stated in papers
    like "attentions is not explainable" are still valid. To be more robust,
    we additionally use gradients and take the mean over model layers
    and heads. Moreover, we provide an exhaustive analysis how accurate this
    pattern recognizer is. Check out details in the README.
    """
    max_patterns: int = 5
    is_scaled: bool = True
    is_rounded: bool = True
    round_decimals: int = 2

    def __call__(
            self,
            example: TokenizedExample,
            output: Output
    ) -> List[Pattern]:
        text_mask = self.text_tokens_mask(example)
        w, pattern_vectors = self.transform(output, text_mask, example.alignment)
        patterns = self.build_patterns(w, example.text_tokens, pattern_vectors)
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

        if self.is_scaled:
            patterns *= w.reshape(-1, 1)
        if self.is_rounded:
            w = np.round(w, decimals=self.round_decimals)
            patterns = np.round(patterns, decimals=self.round_decimals)
        return w, patterns

    @staticmethod
    def text_tokens_mask(example: TokenizedExample) -> List[bool]:
        """ Get the mask of text tokens according to the BERT input. """
        mask = np.zeros(len(example.tokens)).astype(bool)
        mask[1:len(example.text_tokens) + 1] = True
        return mask.tolist()

    def build_patterns(
            self,
            w: np.ndarray,
            tokens: List[str],
            pattern_vectors: np.ndarray
    ) -> List[Pattern]:
        # Negate an array to have a descending order
        indices = np.argsort(w * -1)
        build = lambda i: Pattern(w[i], tokens, pattern_vectors[i, :].tolist())
        return [build(i) for i in indices[:self.max_patterns]]


def predict_key_set(patterns: List[Pattern], n: int) -> Set[int]:
    """ Make sure that patterns before a prediction are scaled by
    importance values. The function returns token indices. """
    weights = np.stack([p.weights for p in patterns]).sum(axis=0)
    decreasing = np.argsort(weights * -1)
    key_set = set(decreasing[:n])
    return key_set
