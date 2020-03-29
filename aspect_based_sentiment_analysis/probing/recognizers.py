from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import List
from typing import Tuple

import numpy as np
import numpy.ma as ma
import tensorflow as tf

from ..alignment import Document
from ..data_types import AspectPattern
from ..data_types import Pattern


class PatternRecognizer(ABC):
    """ The Pattern Recognizer's aim is to discover and
    name patterns which a model uses in a prediction. """

    @abstractmethod
    def __call__(
            self,
            document: Document,
            hidden_states: tf.Tensor,
            attentions: tf.Tensor,
            attention_grads: tf.Tensor
    ) -> Tuple[AspectPattern, List[Pattern]]:
        """ To recognize patterns, we provide detailed information about a
        prediction, including hidden states after each layer, attentions from
        each head in each layer, and attention gradients with respect to the
        model output. The Recognizer returns the Aspect Pattern (the words
        linked to the aspect) and the most significant patterns. """


@dataclass
class AttentionPatternRecognizer(PatternRecognizer):
    """ The Attention Pattern Recognizer uses attentions and their gradients
    to discover the most significant patterns. The key idea is to use
    attention activations and scale them by their gradients (with respect to
    the model output). The language model constructs an enormous amount of
    various relations between words. However, only some of them are crucial
    in the aspect-based sentiment classification. Thanks to gradients,
    we can filter unnecessary patterns out. This heuristic is a rough
    approximation (e.g. we take the mean activation over model heads).
    Nonetheless, it gives a good intuition about how model reasoning works.

    Parameters:
        `percentile_mask` mask weights which are under the weight
        magnitude percentile. Default 80% of lowest weights are wiped off
        (they blur an overall effect).
        `percentile_information` define minimal information amount
        which are in the return patterns. Default 80% of weights magnitude.
    """
    percentile_mask: int = 80
    percentile_information: int = 80

    def __call__(
            self,
            document: Document,
            hidden_states: tf.Tensor,
            attentions: tf.Tensor,
            attention_grads: tf.Tensor
    ) -> Tuple[AspectPattern, List[Pattern]]:
        interest = self.get_interest(attentions, attention_grads)
        patterns = self.get_patterns(document, interest)
        aspect_pattern = self.get_aspect_pattern(document, interest)
        return aspect_pattern, patterns

    def get_interest(
            self,
            attentions: tf.Tensor,
            attention_grads: tf.Tensor
    ) -> np.ndarray:
        """ Calculate the mean of the scaled attentions over model heads,
        called the model `interest`. Mask unnecessary weights. """
        interest = (attentions * attention_grads).numpy()
        interest = np.sum(interest, axis=(0, 1))
        interest = self.clean(interest, percentile=self.percentile_mask)
        return interest

    def get_patterns(
            self,
            document: Document,
            interest: np.ndarray
    ) -> List[Pattern]:
        """ The method tries to discover the most significant patterns.
        Briefly, the model encodes needed information in the class token
        representation and use them to classify the sentiment. Throughout the
        transformer's layers, the model creates contextual word embeddings,
        which we can interpret as the word mixtures. Because of the interest
        includes a gradient part, the first row represents how particular
        mixtures, not words, of the class token representation impact to the
        prediction on average. The approximation of these word `mixtures` are
        rows of the interest matrix. Select only key patterns. """
        cls_id, text_ids, aspect_id = self.get_indices(document)
        impacts = interest[cls_id, text_ids]
        mixtures = interest[text_ids, text_ids]
        key_impacts, key_mixtures = self.get_key_mixtures(
            impacts, mixtures, percentile=self.percentile_information)
        patterns = self.construct_patterns(document, key_impacts, key_mixtures)
        return patterns

    def get_aspect_pattern(
            self,
            document: Document,
            interest: np.ndarray
    ) -> AspectPattern:
        """ The presented sentiment classification is aspect-based, so it is
        worth to know the relation between the aspect and words in the text.
        In this case, we distinguish two sets of weights. As for the other
        patterns, the `aspect_come_from` tells us how each word appeals to
        the aspect representation on average. Also, we add the `aspect_look_at`
        weights to check what it is interesting for the aspect to look at. """
        cls_id, text_ids, aspect_id = self.get_indices(document)
        aspect_come_from = interest[aspect_id, text_ids].tolist()
        aspect_look_at = interest[text_ids, aspect_id].tolist()
        aspect_pattern = AspectPattern(
            document.tokens, aspect_come_from, aspect_look_at)
        return aspect_pattern

    @staticmethod
    def get_indices(document: Document) -> Tuple[int, List[int], int]:
        """ Get indices for the class token, text words, and the aspect word
        according to the BERT input structure. """
        indices = np.arange(len(document.tokens))
        cls_id, *text_ids, sep1_id, aspect_id, sep2_id = indices
        return cls_id, text_ids, aspect_id

    @staticmethod
    def clean(interest: np.ndarray, percentile: int) -> np.ndarray:
        """ Normalize the interest values so that the max magnitude equals
        one. Mask weights which are under the weight magnitude percentile. """
        magnitude = np.abs(interest)
        interest /= magnitude.max()
        threshold = np.percentile(magnitude, percentile)
        mx = ma.masked_array(interest, magnitude < threshold)
        interest = mx.filled(0)
        return interest

    @staticmethod
    def get_key_mixtures(
            impacts: np.ndarray,
            mixtures: np.ndarray,
            percentile: int
    ) -> Tuple[List[float], List[List[float]]]:
        """ Get the most important mixtures, weights of patterns. """
        increasing_order = np.argsort(impacts)
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
            document: Document,
            impacts: List[float],
            mixtures: List[List[float]]
    ) -> List[Pattern]:
        patterns = [Pattern(impact, document.tokens, weights)
                    for impact, weights in zip(impacts, mixtures)]
        return patterns
