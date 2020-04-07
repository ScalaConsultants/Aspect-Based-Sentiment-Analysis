from enum import IntEnum
from dataclasses import dataclass
from typing import Dict
from typing import OrderedDict
from typing import Iterable
from typing import List
from typing import Tuple

import tensorflow as tf


class Sentiment(IntEnum):
    """ We use the three basic polarities to
    classify the aspect based sentiment."""
    neutral = 0
    negative = 1
    positive = 2


@dataclass(frozen=True)
class AspectSpan:
    """ Preprocess the pair of a text (sentence/document)
    and an aspect to the model understandable form.
    The model can not process the raw pair of two
    strings (text, aspect) directly. We need to tokenize
    both at the very beginning. Besides, we have to split
    tokens to sub_tokens using the *word-piece tokenizer*,
    according to the input format of the language model.
    We take care to do the alignment between them for
    better interpretability. For now, only the single
    word should describe an aspect (one token). """
    text: str
    text_tokens: List[str]
    aspect: str
    aspect_tokens: List[str]
    tokens: List[str]
    sub_tokens: List[str]
    alignment: List[List[int]]


@dataclass(frozen=True)
class Pattern:
    """ The weighted tokens describe the `Pattern`.
    Each pattern has a different `impact` to the final
    prediction. """
    impact: float
    tokens: List[str]
    weights: List[float]


@dataclass(frozen=True)
class AspectRepresentation:
    """ The model looks for relations between the text
    and the aspect. The `AspectRepresentation` shows which
    words impact to the final aspect representation
    `come_from` and how the aspect affects different
    word representations `look_at`. """
    tokens: List[str]
    come_from: List[float]
    look_at: List[float]


@dataclass(frozen=True)
class AspectSpanLabeled(AspectSpan):
    """ After the classification, each the Aspect Span
    contains additional attributes such as the sentiment,
    scores for each sentiment class, and patterns. The
    aspect interest and patterns are optional (they are
    if a pipeline has a pattern recognizer). """
    sentiment: Sentiment
    scores: List[float]
    aspect_representation: AspectRepresentation = None
    patterns: List[Pattern] = None


@dataclass(frozen=True)
class AspectDocument:
    """ Due to the computation constraints, before we start
    to preprocess the input pair of strings (text, aspect)
    directly to the `AspectSpan`, we may need to split a long
    text into smaller text chunks, and build several aspect
    spans. They can include a single sentence or several sentences
    (it depends how works the provided sentencizer in a pipeline)
    Please note that longer spans have the richer context information. """
    text: str
    aspect: str
    aspect_spans: List[AspectSpan]

    def __iter__(self) -> Iterable[AspectSpan]:
        return iter(self.aspect_spans)


@dataclass(frozen=True)
class AspectDocumentLabeled(AspectDocument):
    """ This is a container which keeps detailed information
    about the prediction, the labeled aspect spans. It has
    the overall sentiment for the input pair of strings
    (text, aspect). """
    aspect_spans: List[AspectSpanLabeled]
    sentiment: Sentiment
    scores: List[float]


@dataclass(frozen=True)
class Document:
    """ The `Document` collects all pre-processed `AspectDocument`s
    in one place. Commonly, we do the sentiment classification for
    several aspects at the same time. Therefore, to make a prediction,
    we want to stack each aspect-span from each aspect-document
    to the single batch, and finally, pass it to the model. To do so,
    we have the `batch` property. Additionally, the `indices` help to
    convert the batch back to the `DocumentLabeled` after the prediction. """
    text: str
    aspect_docs: OrderedDict[str, AspectDocument]

    @property
    def indices(self) -> List[Tuple[int, int]]:
        indices = []
        start, end = 0, 0
        for aspect_doc in self:
            length = len(list(aspect_doc))
            end += length
            indices.append((start, end))
            start += length
        return indices

    @property
    def batch(self) -> List[AspectSpan]:
        return [span for doc in self for span in doc]

    def __getitem__(self, aspect: str):
        return self.aspect_docs[aspect]

    def __iter__(self) -> Iterable[AspectDocument]:
        aspects = self.aspect_docs.keys()
        return (self[aspect] for aspect in aspects)


@dataclass(frozen=True)
class DocumentLabeled(Document):
    """ The `Document` collects all `AspectDocumentLabeled`s.
    A pipeline returns it as the final prediction result. """
    aspect_docs: Dict[str, AspectDocumentLabeled]


@dataclass(frozen=True)
class InputBatch:
    """ The model uses these tensors to perform a prediction.
    The names are compatible with the *transformers* package.
    The `token_ids` contains indices of input sequence
    _sub_tokens_ in the vocabulary. The `attention_mask` is used
    to avoid performing attention on padding token indices
    (this is not related with masks from the language modeling
    task). The `token_type_ids` is a segment token indices
    to indicate first and second portions of the inputs,
    zeros and ones. """
    token_ids: tf.Tensor
    attention_mask: tf.Tensor
    token_type_ids: tf.Tensor


@dataclass(frozen=True)
class OutputBatch:
    """ The model returns not only scores, the softmax of logits,
    but also stacked hidden states [batch, layer, sequence, embedding],
    attentions [batch, layer, head, attention, attention], and attention
    gradients with respect to the model output. All of them are useful
    not only for the classification, but also we use them to probe and
    understand model's decision-making. """
    scores: tf.Tensor
    hidden_states: tf.Tensor
    attentions: tf.Tensor
    attention_grads: tf.Tensor
