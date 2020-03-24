from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Tuple
from typing import List
from typing import Optional

import numpy as np
import tensorflow as tf
import transformers

from . import utils
from .data_types import Prediction
from .data_types import Sentiment
from .data_types import ClassifierExample
from .models import BertABSClassifier


class Pipeline(ABC):
    """ The pipeline provides the affordable interface for making
    predictions using the fine-tuned model along with the tokenizer. """

    @abstractmethod
    def __call__(self, text: str, aspects: List[str]) -> List[Prediction]:
        """ Perform Aspect Based Sentiment Classification. You can check
        several aspects at once. """


@dataclass(frozen=True)
class BertPipeline(Pipeline):
    model: BertABSClassifier
    tokenizer: transformers.BertTokenizer

    def __call__(self, text: str, aspects: List[str]) -> List[Prediction]:
        pairs = [(text, aspect) for aspect in aspects]
        predictions, logits, *details = self.predict(pairs)
        return predictions

    def predict(
            self,
            pairs: List[Tuple[str, str]]
    ) -> Tuple[List[Prediction], tf.Tensor, Optional[List[Tuple[tf.Tensor]]]]:
        """ Each pair represents (text, aspect). """
        encoded = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=pairs,
            add_special_tokens=True,
            pad_to_max_length='right',
            return_tensors='tf'
        )
        logits, *details = self.model.call(
            token_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask'],
            token_type_ids=encoded['token_type_ids']
        )
        batch_scores = tf.nn.softmax(logits, axis=1).numpy()
        predictions = [self.build_prediction(pair, scores)
                       for pair, scores in zip(pairs, batch_scores)]
        return (predictions, logits, *details)

    def evaluate(self,
                 examples: List[ClassifierExample],
                 metric: tf.metrics.Metric,
                 batch_size: int = 32) -> np.ndarray:
        batches = utils.batches(examples, batch_size)
        for batch in batches:
            pairs = [(e.text, e.aspect) for e in batch]
            predictions, logits, *details = self.predict(pairs)
            y_pred = [p.sentiment.value for p in predictions]
            y_true = [e.sentiment.value for e in batch]
            metric.update_state(y_true, y_pred)
        result = metric.result().numpy()
        return result

    @staticmethod
    def build_prediction(pair: Tuple[str, str],
                         scores: np.ndarray) -> Prediction:
        text, aspect = pair
        sentiment_id = scores.argmax().astype(int)
        scores = scores.tolist()
        sentiment = Sentiment(sentiment_id)
        prediction = Prediction(aspect, sentiment, text, scores)
        return prediction
