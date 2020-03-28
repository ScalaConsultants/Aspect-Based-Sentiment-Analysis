from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import List
from typing import Tuple

import numpy as np
import numpy.ma as ma
import tensorflow as tf

from ..alignment import Document
from ..data_types import Pattern


class Checker(ABC):

    @abstractmethod
    def __call__(
            self,
            document: Document,
            hidden_states: tf.Tensor,
            attentions: tf.Tensor,
            attention_grads: tf.Tensor
    ) -> List[Pattern]:
        """ """


@dataclass
class InterestChecker(Checker):
    pattern_name: str = 'interest'
    coreference_name: str = 'interest-coreference'
    mask_percentile: int = 80
    magnitude_threshold: float = 0.80

    def __call__(
            self,
            document: Document,
            hidden_states: tf.Tensor,
            attentions: tf.Tensor,
            attention_grads: tf.Tensor
    ) -> List[Pattern]:
        # Note that one word should describe an aspect.
        tokens = document.tokens
        indices = np.arange(len(document.tokens))
        cls_id, *text_ids, sep1_id, aspect_id, sep2_id = indices

        interest = (attentions * attention_grads).numpy()
        interest = np.sum(interest, axis=(0, 1))
        interest = self.normalize(interest)

        # The sentiment prediction directly depends on the final class
        # token embedding.
        #
        #
        # The first row tells us how approximately it is built.
        # structures of word mixtures
        impacts = interest[cls_id, text_ids]
        mixtures = interest[:, text_ids]
        key_mixtures = self.get_key_mixtures(impacts, mixtures)

        coreference = interest[aspect_id, text_ids]
        patterns = self.get_patterns(tokens, key_mixtures, coreference)
        return patterns

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """ """
        magnitude = np.abs(x)
        x = (x / magnitude.max()).astype(int)
        threshold = np.percentile(magnitude, self.mask_percentile)
        mx = ma.masked_array(x, magnitude < threshold)
        x = mx.filled(0)
        return x

    def get_key_mixtures(
            self,
            impacts: np.ndarray,
            mixtures: np.ndarray
    ) -> List[Tuple[float, List[float]]]:
        """ """
        increasing_order = np.argsort(impacts)
        order = increasing_order[::-1]
        total_magnitude = np.sum(np.abs(impacts))
        magnitude = 0
        results = []
        for i in order:
            impact = impacts[i]
            weights = mixtures[i].tolist()
            results.append((impact, weights))
            magnitude += np.abs(impact)
            if magnitude / total_magnitude >= self.magnitude_threshold:
                return results

    def get_patterns(
            self,
            tokens: List[str],
            mixtures: [Tuple[float, List[float]]],
            coreference: np.ndarray
    ) -> List[Pattern]:
        """ """
        patterns = [Pattern(self.pattern_name, impact, tokens, weights)
                    for impact, weights in mixtures]
        # We add the coreference pattern which describes relation
        # between the aspect and words in a text.
        coref_pattern = Pattern(name=self.coreference_name,
                                impact=sum(coreference),
                                tokens=tokens,
                                weights=list(coreference))
        patterns.append(coref_pattern)
        return patterns
