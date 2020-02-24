from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Iterable
from typing import List

import tensorflow as tf
import transformers

from .data_types import AspectPrediction
from .models import BertABSClassifier


class Pipeline(ABC):
    """ """

    @abstractmethod
    def __call__(self, text: str, aspects: List[str] = None) -> Iterable[AspectPrediction]:
        """ """


@dataclass(frozen=True)
class BertPipeline(Pipeline):
    model: BertABSClassifier
    tokenizer: transformers.BertTokenizer

    def __call__(self, text: str, aspect_names: List[str] = None) -> List[AspectPrediction]:
        return self.predict(text, aspect_names) if aspect_names \
               else self.extract_and_predict(text)

    def extract_and_predict(self, text: str) -> List[AspectPrediction]:
        """ """
        encoded = self.tokenizer.batch_encode_plus(
            [text],
            add_special_tokens=True,
            pad_to_max_length='right',
            return_tensors='tf'
        )
        outputs = self.model.call(
            token_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask'],
            token_type_ids=encoded['token_type_ids']
        )
        extractor_logits, extractor_details, \
        classifier_logits, classifier_details = outputs
        return self.build_aspect_predictions(aspect_names, classifier_logits, text)

    def predict(self, text: str, aspect_names: List[str]) -> List[AspectPrediction]:
        """ """
        text_pairs = [(text, aspect_name) for aspect_name in aspect_names]
        encoded = self.tokenizer.batch_encode_plus(
            text_pairs,
            add_special_tokens=True,
            pad_to_max_length='right',
            return_tensors='tf'
        )
        classifier_logits, *details = self.model.call_classifier(
            token_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask'],
            token_type_ids=encoded['token_type_ids']
        )
        return self.build_aspect_predictions(aspect_names, classifier_logits, text)

    @staticmethod
    def build_aspect_predictions(names, logits, text):
        """ """
        scores = tf.nn.softmax(logits, axis=1).numpy()
        labels = [int(score) for score in scores]
        return [AspectPrediction(name, label, text, score)
                for name, label, score in zip(names, labels, scores)]
