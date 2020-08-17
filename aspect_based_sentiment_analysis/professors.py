from abc import ABC
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from .aux_models import ReferenceRecognizer
from .aux_models import PatternRecognizer
from .data_types import Iterable
from .data_types import PredictedExample
from .data_types import OutputBatch
from .data_types import Review
from .data_types import Sentiment
from .data_types import Task


@dataclass
class _Professor(ABC):
    """ """
    ref_recognizer: ReferenceRecognizer = None
    pattern_recognizer: PatternRecognizer = None

    def make_decision(
            self,
            task: Task,
            output_batch: OutputBatch
    ) -> Iterable[PredictedExample]:
        """ """


@dataclass
class Professor(_Professor):

    def make_decision(
            self,
            task: Task,
            output_batch: OutputBatch
    ) -> Iterable[PredictedExample]:
        reviews = self.review(task, output_batch)
        originals = self.get_originals(output_batch.scores)
        for example, original, review in zip(task, originals, reviews):

            if review.is_reference is False:
                sentiment = Sentiment.neutral
                scores = [0, 0, 0]
            else:
                sentiment, scores = original

            prediction = PredictedExample.from_example(
                example, sentiment=sentiment, scores=scores, review=review)
            yield prediction

    def review(
            self,
            task: Task,
            output_batch: OutputBatch
    ) -> Iterable[Review]:
        for example, args in zip(task, output_batch):
            is_reference = self.ref_recognizer(example, *args) \
                if self.ref_recognizer else None
            patterns = self.pattern_recognizer(example, *args) \
                if self.pattern_recognizer and is_reference is not False else\
                None
            review = Review(is_reference, patterns)
            yield review

    @staticmethod
    def get_originals(batch_scores: tf.Tensor) -> Iterable[
        Tuple[Sentiment, List[float]]]:
        for scores in batch_scores:
            sentiment_id = np.argmax(scores).astype(int)
            sentiment = Sentiment(sentiment_id)
            scores = list(scores)
            yield sentiment, scores
