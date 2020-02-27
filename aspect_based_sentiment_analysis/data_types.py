from abc import ABC
from enum import IntEnum
from dataclasses import dataclass
from typing import List


class Label(IntEnum):
    neutral = 0
    negative = 1
    positive = 2


@dataclass(frozen=True)
class Aspect:
    name: str
    label: Label


@dataclass(frozen=True)
class AspectPrediction(Aspect):
    text: str
    scores: float = None
    confidence: float = None

    def __str__(self):
        return f'Aspect(name={self.name}, label={self.label})'


class Example(ABC):
    """ """


@dataclass(frozen=True)
class LanguageModelExample(Example):
    text_a: str
    text_b: str
    is_next: int


@dataclass(frozen=True)
class ExtractorExample(Example):
    text: str
    aspect_names: List[str]


@dataclass(frozen=True)
class ClassifierExample(Example):
    text: str
    aspect: Aspect


@dataclass(frozen=True)
class MultimodalExample(Example):
    text: str
    aspects: List[Aspect]
