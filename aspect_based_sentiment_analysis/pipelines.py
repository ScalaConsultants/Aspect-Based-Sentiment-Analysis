import logging
from abc import ABC
from abc import abstractmethod
from collections import OrderedDict
from dataclasses import asdict
from dataclasses import dataclass
from typing import Callable
from typing import Iterable
from typing import List
from typing import Tuple

import numpy as np
import tensorflow as tf
import transformers

from . import alignment
from . import utils
from .data_types import AspectSpan
from .data_types import AspectSpanLabeled
from .data_types import AspectDocument
from .data_types import AspectDocumentLabeled
from .data_types import Document
from .data_types import DocumentLabeled
from .data_types import InputBatch
from .data_types import OutputBatch
from .data_types import Sentiment
from .models import ABSClassifier
from .models import BertABSClassifier
from .probing import PatternRecognizer
from .training import ClassifierExample
from .training import classifier_loss

logger = logging.getLogger('absa.pipeline')


@dataclass
class Pipeline(ABC):
    """
    The pipeline simplify the use of the fine-tuned Aspect-Based
    Sentiment Classifier.

    For the basic inference, you benefit from the `__call__` method. You just
    pass the raw document text together with aspects. The pipeline performs
    several clear transformations:
        - convert raw text and aspects into the document,
        - stack each aspect-spans from the document, and
          encode them into the model compatible input batch,
        - pass it to the model, and get the output batch,
        - use the output batch, to label the aspect-spans,
          and build the labeled document,
    which the pipeline returns to you. If you wish to build your own pipeline,
    we hope you should benefit from some of them.

    The aim is to classify the sentiment of a potentially long text for
    several aspects. We made two important design decisions. Firstly,
    even some research presents how to make a prediction for several aspects
    at once, we process aspects independently. Secondly, we split a text into
    smaller independent chunks, called spans. They can include a single
    sentence or several sentences. It depends how works a `sentencizer`. Note
    that longer spans have the richer context information.

    Equally importantly, the pipeline interprets the model predictions.
    Thanks to the integrated `pattern_recognizer`, we can investigate how
    much results are reliable. In our task, we are curious about two thinks
    at most. Firstly, we want to be sure that the model connects the correct
    word or words with the aspect. If the model does it wrong, the sentiment
    concerns the different entity. Secondly, even if the model recognized the
    aspect correctly, we wish to better understand the model reasoning. To do
    so, the pattern recognizer estimates the key patterns, the weighted
    sequence of words, and theirs approximated impact to the prediction. We
    want to avoid situation wherein a single word or weird word's combination
    triggers the model.

    Please note that the package contains the separated submodule
    `absa.training`. You can find there complete routines to tune either the
    language model or the classification layer. Check out the examples on the
    home repository.
    """
    model: ABSClassifier
    tokenizer: transformers.PreTrainedTokenizer
    sentencizer: Callable[[str], List[str]]
    pattern_recognizer: PatternRecognizer

    @abstractmethod
    def __call__(self, text: str, aspects: List[str]) -> DocumentLabeled:
        """


        Parameters
        ----------
        text

        aspects

        Returns
        -------
        doc_labeled

        """

    @abstractmethod
    def get_document(self, text: str, aspects: List[str]) -> Document:
        """"""

    @abstractmethod
    def preprocess(self, pairs: List[Tuple[str, str]]) -> List[AspectSpan]:
        """ """

    @abstractmethod
    def encode(self, aspect_spans: List[AspectSpan]) -> InputBatch:
        """ """

    @abstractmethod
    def predict(self, batch: InputBatch) -> OutputBatch:
        """ """

    @abstractmethod
    def label(
            self,
            aspect_spans: List[AspectSpan],
            batch: OutputBatch
    ) -> List[AspectSpanLabeled]:
        """ """

    @abstractmethod
    def evaluate(
            self,
            examples: List[ClassifierExample],
            metric: tf.metrics.Metric,
            batch_size: int
    ) -> tf.Tensor:
        """ """


@dataclass
class BertPipeline(Pipeline):
    model: BertABSClassifier
    tokenizer: transformers.BertTokenizer
    sentencizer: Callable[[str], List[str]] = None
    pattern_recognizer: PatternRecognizer = None

    def __call__(
            self,
            text: str,
            aspects: List[str],
            skip_probe: bool = False
    ) -> DocumentLabeled:
        doc = self.get_document(text, aspects)
        aspect_spans = doc.batch
        input_batch = self.encode(aspect_spans)
        output_batch = self.predict(input_batch)
        aspect_spans_labeled = self.label(aspect_spans, output_batch)
        doc_labeled = self.get_document_labeled(doc, aspect_spans_labeled)
        return doc_labeled

    def get_document(self, text: str, aspects: List[str]) -> Document:
        texts = self.sentencizer(text) if self.sentencizer else [text]
        aspect_docs = OrderedDict()
        for aspect in aspects:
            pairs = [(text, aspect) for text in texts]
            aspect_spans = self.preprocess(pairs)
            aspect_doc = AspectDocument(text, aspect, aspect_spans)
            aspect_docs[aspect] = aspect_doc
        document = Document(text, aspect_docs)
        return document

    def preprocess(self, pairs: List[Tuple[str, str]]) -> List[AspectSpan]:
        aspect_spans = [
            alignment.make_aspect_span(self.tokenizer, text, aspect)
            for text, aspect in pairs
        ]
        return aspect_spans

    def encode(self, aspect_spans: List[AspectSpan]) -> InputBatch:
        token_pairs = [(aspect_span.text_tokens, aspect_span.aspect_tokens)
                       for aspect_span in aspect_spans]
        encoded = self.tokenizer.batch_encode_plus(
            token_pairs,
            add_special_tokens=True,
            pad_to_max_length='right',
            return_tensors='tf'
        )
        batch = InputBatch(
            token_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask'],
            token_type_ids=encoded['token_type_ids']
        )
        return batch

    def predict(self, batch: InputBatch) -> OutputBatch:
        # This implementation forces the model to return the detailed
        # output including hidden states and attentions.
        with tf.GradientTape() as tape:
            logits, hidden_states, attentions = self.model.call(
                token_ids=batch.token_ids,
                attention_mask=batch.attention_mask,
                token_type_ids=batch.token_type_ids
            )
            # We assume that our predictions are correct. This is
            # required to calculate the attention gradients for
            # probing and exploratory purposes.
            predictions = np.argmax(logits, axis=-1)
            labels = tf.one_hot(predictions, depth=3)
            loss_value = classifier_loss(labels, logits)
        attention_grads = tape.gradient(loss_value, attentions)

        # Compute the final prediction scores.
        scores = tf.nn.softmax(logits, axis=1)

        # Stack a tensor tuple into a single multi-dim array:
        #   hidden states: [batch, layer, sequence, embedding]
        #   attentions: [batch, layer, head, attention, attention]
        # Note that we make an assumption that the embedding's size
        # is the same as the model hidden states.
        stack = lambda x, order: tf.transpose(tf.stack(x), order)
        hidden_states = stack(hidden_states, [1, 0, 2, 3])
        attentions = stack(attentions, [1, 0, 2, 3, 4])
        attention_grads = stack(attention_grads, [1, 0, 2, 3, 4])
        output_batch = OutputBatch(
            scores=scores,
            hidden_states=hidden_states,
            attentions=attentions,
            attention_grads=attention_grads
        )
        return output_batch

    def label(
            self,
            aspect_spans: List[AspectSpan],
            batch: OutputBatch
    ) -> Iterable[AspectSpanLabeled]:
        sentiment_ids = np.argmax(batch.scores, axis=-1).astype(int)
        for i, aspect_span in enumerate(aspect_spans):
            sentiment_id = sentiment_ids[i]
            aspect_representation, patterns = self.pattern_recognizer(
                aspect_span=aspect_span,
                hidden_states=batch.hidden_states[i],
                attentions=batch.attentions[i],
                attention_grads=batch.attention_grads[i]
            ) if self.pattern_recognizer else (None, None)
            kwargs = asdict(aspect_span)
            aspect_span_labeled = AspectSpanLabeled(
                sentiment=Sentiment(sentiment_id),
                scores=batch.scores[i].numpy().tolist(),
                aspect_representation=aspect_representation,
                patterns=patterns,
                **kwargs
            )
            yield aspect_span_labeled

    def evaluate(
            self,
            examples: List[ClassifierExample],
            metric: tf.metrics.Metric,
            batch_size: int
    ) -> tf.Tensor:
        batches = utils.batches(examples, batch_size)
        for batch in batches:
            pairs = [(e.text, e.aspect) for e in batch]
            aspect_spans = self.preprocess(pairs)
            input_batch = self.encode(aspect_spans)
            output_batch = self.predict(input_batch)
            aspect_span_labeled = self.label(aspect_spans, output_batch)
            y_pred = [a.sentiment.value for a in aspect_span_labeled]
            y_true = [e.sentiment.value for e in batch]
            metric.update_state(y_true, y_pred)
        result = metric.result()
        return result

    @staticmethod
    def get_document_labeled(
            document: Document,
            batch: Iterable[AspectSpanLabeled]
    ) -> DocumentLabeled:
        batch = list(batch)
        aspect_documents = OrderedDict()
        for start, end in document.indices:
            aspect_spans = batch[start:end]
            first = aspect_spans[0]
            scores = np.max([s.scores for s in aspect_spans], axis=0)
            scores /= np.linalg.norm(scores, ord=1)
            sentiment_id = np.argmax(scores).astype(int)
            aspect_document = AspectDocumentLabeled(
                text=document.text,
                aspect=first.aspect,
                aspect_spans=aspect_spans,
                sentiment=Sentiment(sentiment_id),
                scores=list(scores)
            )
            aspect_documents[first.aspect] = aspect_document
        document = DocumentLabeled(document.text, aspect_documents)
        return document
