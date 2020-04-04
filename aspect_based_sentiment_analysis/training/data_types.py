from abc import ABC
from dataclasses import dataclass
from .. import Sentiment


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
