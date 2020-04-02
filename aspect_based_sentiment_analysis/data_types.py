from abc import ABC
from enum import IntEnum
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Iterable

import tensorflow as tf


class Sentiment(IntEnum):
    """ We use the three basic polarities to
    classify the aspect based sentiment."""
    neutral = 0
    negative = 1
    positive = 2


@dataclass(frozen=True)
class Pattern:
    """ The weighted tokens describes the `Pattern`.
    Each pattern has a different `impact` to the final
    prediction. """
    impact: float
    tokens: List[str]
    weights: List[float]


@dataclass(frozen=True)
class AspectInterest:
    """  """
    tokens: List[str]
    come_from: List[float]
    look_at: List[float]


@dataclass(frozen=True)
class AspectSpan:
    """ The single word should describe
    an aspect (one token)."""
    text: str
    text_tokens: List[str]
    aspect: str
    aspect_tokens: List[str]
    tokens: List[str]
    sub_tokens: List[str]
    alignment: List[List[int]]


@dataclass(frozen=True)
class AspectSpanLabeled(AspectSpan):
    sentiment: Sentiment
    scores: List[float]
    aspect_interest: AspectInterest = None
    patterns: List[Pattern] = None


@dataclass(frozen=True)
class AspectDocument:
    text: str
    aspect: str
    aspect_spans: List[AspectSpan]

    def __iter__(self) -> Iterable[AspectSpan]:
        return self.aspect_spans


@dataclass(frozen=True)
class AspectDocumentLabeled(AspectDocument):
    aspect_spans: List[AspectSpanLabeled]
    sentiment: Sentiment
    scores: List[float]


@dataclass(frozen=True)
class InputBatch:
    token_ids: tf.Tensor
    attention_mask: tf.Tensor
    token_type_ids: tf.Tensor


@dataclass(frozen=True)
class OutputBatch:
    scores: tf.Tensor
    hidden_states: tf.Tensor
    attentions: tf.Tensor
    attention_grads: tf.Tensor


@dataclass(frozen=True)
class Document:
    text: str
    aspect_documents: OrderedDict[str, AspectDocument]

    @property
    def flatten(self) -> List[AspectSpan]:
        return [span for doc in self for span in doc]

    def __getitem__(self, aspect: str):
        return self.aspect_documents[aspect]

    def __iter__(self) -> Iterable[AspectDocument]:
        aspects = self.aspect_documents.keys()
        return (self[aspect] for aspect in aspects)


@dataclass(frozen=True)
class DocumentLabeled(Document):
    aspect_documents: Dict[str, AspectDocumentLabeled]


class TrainExample(ABC):
    """ The Train Example represents a single
     observation used in the training. """


@dataclass(frozen=True)
class LanguageModelExample(TrainExample):
    text_a: str
    text_b: str
    is_next: int


@dataclass(frozen=True)
class ClassifierExample(TrainExample):
    text: str
    aspect: str
    sentiment: Sentiment
