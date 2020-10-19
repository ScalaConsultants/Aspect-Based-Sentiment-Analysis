from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass

import numpy as np

from .aux_models import ReferenceRecognizer
from .aux_models import PatternRecognizer
from .data_types import TokenizedExample
from .data_types import PredictedExample
from .data_types import Output
from .data_types import Review
from .data_types import Sentiment


@dataclass
class _Professor(ABC):

    @abstractmethod
    def review(
            self,
            example: TokenizedExample,
            output_batch: Output
    ) -> PredictedExample:
        pass


@dataclass
class Professor(_Professor):
    reference_recognizer: ReferenceRecognizer = None
    pattern_recognizer: PatternRecognizer = None

    def review(
            self,
            example: TokenizedExample,
            output: Output
    ) -> PredictedExample:
        scores = list(output.scores.numpy())
        sentiment_id = np.argmax(scores).astype(int)
        sentiment = Sentiment(sentiment_id)

        is_reference = self.reference_recognizer(example, output) \
            if self.reference_recognizer else None
        patterns = self.pattern_recognizer(example, output) \
            if self.pattern_recognizer and is_reference is not False else None
        review = Review(is_reference, patterns)

        if review.is_reference is False:
            sentiment = Sentiment.neutral
            scores = [0, 0, 0]

        prediction = PredictedExample.from_example(
            example, sentiment=sentiment, scores=scores, review=review)
        return prediction
