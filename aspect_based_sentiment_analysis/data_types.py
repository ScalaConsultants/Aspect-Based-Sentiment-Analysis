from abc import ABC
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Aspect:
    name: str
    label: int


@dataclass(frozen=True)
class AspectPrediction(Aspect):
    text: str
    score: float = None
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
