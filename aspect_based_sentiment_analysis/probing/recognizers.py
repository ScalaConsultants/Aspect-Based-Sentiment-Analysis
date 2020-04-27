from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import List
from typing import Tuple

import numpy as np
import tensorflow as tf

from ..data_types import TokenizedExample
from ..data_types import AspectRepresentation
from ..data_types import Pattern


class PatternRecognizer(ABC):
    """ The Pattern Recognizer's aim is to discover and
    name patterns which a model uses in a prediction. """

    @abstractmethod
    def __call__(
            self,
            example: TokenizedExample,
            hidden_states: tf.Tensor,
            attentions: tf.Tensor,
            attention_grads: tf.Tensor
    ) -> Tuple[AspectRepresentation, List[Pattern]]:
        """ To recognize patterns, we provide detailed information about a
        prediction, including hidden states after each layer, attentions from
        each head in each layer, and attention gradients with respect to the
        model output. The Recognizer returns the aspect representation (the
        words related to the aspect) and the most significant patterns. """


@dataclass
class AttentionGradientProduct(PatternRecognizer):
    """ The Attention Gradient Product uses attentions and their gradients to
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

    Parameters:
        `information_in_patterns` returns the key patterns which coverts the
        percentile of the total information. Default 80% of weights magnitude.
    """
    information_in_patterns: int = 80

    def __call__(
            self,
            example: TokenizedExample,
            hidden_states: tf.Tensor,
            attentions: tf.Tensor,
            attention_grads: tf.Tensor
    ) -> Tuple[AspectRepresentation, List[Pattern]]:
        product = self.get_product(attentions, attention_grads)
        patterns = self.get_patterns(example, product)
        aspect = self.get_aspect_representation(example, product)
        return aspect, patterns

    @staticmethod
    def get_product(
            attentions: tf.Tensor,
            attention_grads: tf.Tensor
    ) -> np.ndarray:
        """ Calculate the attention-gradient product. Take the mean
        over model layers and heads. """
        product = (attentions * attention_grads).numpy()
        product = np.sum(product, axis=(0, 1))
        return product

    def get_patterns(
            self,
            example: TokenizedExample,
            product: np.ndarray
    ) -> List[Pattern]:
        """ The method tries to discover the most significant patterns.
        Briefly, the model encodes needed information in the class token
        representation and use them to classify the sentiment. Throughout the
        transformer's layers, the model creates contextual word embeddings,
        which we can interpret as the word mixtures. Because of the `product`
        includes a gradient part, the first row represents how particular
        mixtures, not words, of the class token representation impact to the
        prediction on average. The approximation of these word `mixtures` are
        rows of the product matrix. Select only key patterns. """
        cls_id, text_ids, aspect_id = self.get_indices(example)
        # Note that the gradient comes from the loss function, and it is why
        # we have to change the sign to get a direction of the improvement.
        impacts = product[cls_id, text_ids] * -1
        mixtures = np.abs(product[text_ids, :][:, text_ids])
        impacts = self.scale(impacts)
        mixtures = self.scale(mixtures)
        key_impacts, key_mixtures = self.get_key_mixtures(
            impacts, mixtures, percentile=self.information_in_patterns)
        patterns = self.construct_patterns(example, key_impacts, key_mixtures)
        return patterns

    def get_aspect_representation(
            self,
            example: TokenizedExample,
            product: np.ndarray
    ) -> AspectRepresentation:
        """ The presented sentiment classification is aspect-based, so it is
        worth to know the relation between the aspect and words in the text.
        In this case, we distinguish two sets of weights. As for the other
        patterns, the `come_from` tells us how each word appeals to the
        aspect representation on average. Also, we add the `look_at` weights
        to check what it is interesting for the aspect to look at. """
        cls_id, text_ids, aspect_id = self.get_indices(example)
        come_from = np.abs(product[aspect_id, text_ids])
        look_at = np.abs(product[text_ids, aspect_id])
        come_from = self.scale(come_from).tolist()
        look_at = self.scale(look_at).tolist()
        aspect_representation = AspectRepresentation(
            example.text_tokens, come_from, look_at)
        return aspect_representation

    @staticmethod
    def get_indices(example: TokenizedExample) -> Tuple[int, List[int], int]:
        """ Get indices for the class token, text words, and the aspect word
        according to the BERT input structure. """
        indices = np.arange(len(example.tokens))
        cls_id, *text_ids, sep1_id, aspect_id, sep2_id = indices
        return cls_id, text_ids, aspect_id

    @staticmethod
    def scale(x: np.ndarray, epsilon: float = 1e-16) -> np.ndarray:
        """ Scale the array so that the max magnitude equals one. """
        scaled = x / np.max(np.abs(x) + epsilon)
        return scaled

    @staticmethod
    def get_key_mixtures(
            impacts: np.ndarray,
            mixtures: np.ndarray,
            percentile: int
    ) -> Tuple[List[float], List[List[float]]]:
        """ Get the most important mixtures, weights of patterns. """
        increasing_order = np.argsort(np.abs(impacts))
        order = increasing_order[::-1]
        results = []
        magnitude = 0
        total_magnitude = np.sum(np.abs(impacts))
        for i in order:
            impact = impacts[i]
            weights = mixtures[i].tolist()
            results.append((impact, weights))
            magnitude += np.abs(impact)
            if magnitude / total_magnitude * 100 >= percentile:
                key_impacts, key_mixtures = zip(*results)
                return key_impacts, key_mixtures

    @staticmethod
    def construct_patterns(
            example: TokenizedExample,
            impacts: List[float],
            mixtures: List[List[float]]
    ) -> List[Pattern]:
        patterns = [Pattern(impact, example.text_tokens, weights)
                    for impact, weights in zip(impacts, mixtures)]
        return patterns
