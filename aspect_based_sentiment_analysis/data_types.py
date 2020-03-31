from abc import ABC
from enum import IntEnum
from dataclasses import dataclass
from typing import List


class Sentiment(IntEnum):
    neutral = 0
    negative = 1
    positive = 2


@dataclass(frozen=True)
class Pattern:
    impact: float
    tokens: List[str]
    weights: List[float]


@dataclass(frozen=True)
class AspectPattern:
    tokens: List[str]
    aspect_come_from: List[float]
    aspect_look_at: List[float]


@dataclass(frozen=True)
class Prediction:
    text: str
    aspect: str
    sentiment: Sentiment
    scores: List[float]
    aspect_pattern: AspectPattern = None
    patterns: List[Pattern] = None


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
