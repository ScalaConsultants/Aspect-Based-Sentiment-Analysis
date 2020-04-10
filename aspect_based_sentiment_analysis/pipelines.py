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
    The pipeline simplifies the use of the fine-tuned Aspect-Based Sentiment
    Classifier. The aim is to classify the sentiment of a potentially long
    text for several aspects. Furthermore, the pipeline gives the reasons for
    a decision, so we can infer how much results are reliable. For the basic
    inference, you benefit from the `__call__` method.

    We made two important design decisions. Firstly, even some research
    presents how predict several aspects at once, we process aspects
    independently. The used model does not support the multi-aspect
    prediction. Secondly, the sentiment of a long text tends to be fuzzy and
    neutral. Therefore, we split a text into smaller independent chunks,
    called spans. They can include a single sentence or several sentences.
    It depends how works a `text_splitter`. Note that longer spans have
    richer context information, but they requires significant computation
    resources.

    Thanks to the integrated `pattern_recognizer`, we can investigate how
    much predictions are reliable. In our task, we are curious about two
    things at most. Firstly, we want to be sure that the model connects the
    correct word or words with the aspect. If the model does it wrong,
    the sentiment concerns the different entities. Secondly, even if the
    model recognized the aspect correctly, we wish to understand the model
    reasoning better. To do so, the pattern recognizer discovers patterns,
    a weighted sequence of words, and their approximated impact to the
    prediction. We want to avoid a situation wherein a single word or weird
    word combination triggers the model.

    Please note that the package contains the separated submodule
    `absa.training`. You can find there complete routines to tune or train
    either the language model or the classifier. Check out the examples on
    the package website.
    """
    model: ABSClassifier
    tokenizer: transformers.PreTrainedTokenizer
    text_splitter: Callable[[str], List[str]]
    pattern_recognizer: PatternRecognizer

    @abstractmethod
    def __call__(self, text: str, aspects: List[str]) -> DocumentLabeled:
        """
        The __call__ method is for the basic inference. The pipeline
        performs several clear transformations:
            - convert raw text and aspects into the document,
            - stack each aspect-spans from the document, and
              encode them into the model compatible input input_batch,
            - pass it to the model, and get the output input_batch,
            - use the output input_batch, to label the aspect-spans,
              and build the labeled document,
        which the pipeline returns to you.

        Parameters
        ----------
        text
            This is the raw document text without rigorous length limit.
            The text_splitter, if provided, splits the text into smaller spans,
            and the pipeline processes them independently.
        aspects
            The aspects for which the pipeline does the sentiment analysis.
            For now, only the single word should describe an aspect
            (one token, do not combine words using a hyphen).

        Returns
        -------
        doc_labeled
            The labeled document aggregates partial results, labeled aspect
            spans, into one data structure.
        """

    @abstractmethod
    def get_document(self, text: str, aspects: List[str]) -> Document:
        """
        Preprocess the raw document text and aspects. The document is a
        container of aspect-document pairs where each aspect-document
        collects aspect-spans. The aspect-span is an independent
        *preprocessed* sample which we further encode and pass to the model.

        Note that the span can include a single sentence or several
        sentences. It depends how works a `text_splitter`.

        Parameters
        ----------
        text
            This is the raw document text without rigorous length limit.
            However, if the pipeline has the text_splitter, it splits a text
            into smaller chunks, called spans.
        aspects
            The aspects for which the pipeline does the sentiment analysis.

        Returns
        -------
        document
            The document collects pre-processed aspect-documents.
        """

    @abstractmethod
    def preprocess(self, pairs: List[Tuple[str, str]]) -> List[AspectSpan]:
        """
        Preprocess pairs of a text and an aspect into aspect-spans,
        the independent preprocessed sample. The model can not process the
        raw pair of two strings (text, aspect) directly. We need to tokenize
        both at the very beginning.

        Parameters
        ----------
        pairs
            The list of the (text, aspect) string pairs.

        Returns
        -------
        aspect_spans
            The independent preprocessed aspect-span samples.
        """

    @abstractmethod
    def encode(self, aspect_spans: List[AspectSpan]) -> InputBatch:
        """
        Encode preprocessed aspect-spans. The input input_batch is a
        container of tensors crucial for the model to make a prediction.
        The names are compatible with the *transformers* package.

        Parameters
        ----------
        aspect_spans
            The independent preprocessed aspect-span samples.

        Returns
        -------
        input_batch
            The container of tensors needed to make a prediction.
        """

    @abstractmethod
    def predict(self, input_batch: InputBatch) -> OutputBatch:
        """
        Pass the input output_batch to the pretrained model to make a
        prediction.
        The pipeline collects not only scores, the softmax of logits,
        but also hidden states, attentions, and attention gradients with
        respect to the model output. In the end, the method packs them into
        the output output_batch and returns.

        Parameters
        ----------
        input_batch
            The container of tensors needed to make a prediction.

        Returns
        -------
        output_batch
            The container of tensors describing a prediction.
        """

    @abstractmethod
    def label(
            self,
            aspect_spans: List[AspectSpan],
            output_batch: OutputBatch
    ) -> List[AspectSpanLabeled]:
        """
        Label aspect spans using the detailed information about the
        prediction. The aspect-span-labeled contains additional attributes
        such as the sentiment and scores for each sentiment class. The aspect
        interest and patterns are optional. They are if a pipeline has a
        **pattern recognizer**.

        Parameters
        ----------
        aspect_spans
            The independent preprocessed aspect-span samples.

        output_batch
            The container of tensors describing a prediction.

        Returns
        -------
        aspect_spans_labeled
            The list of labeled aspect-span samples.
        """

    @abstractmethod
    def evaluate(
            self,
            examples: List[ClassifierExample],
            metric: tf.metrics.Metric,
            batch_size: int
    ) -> tf.Tensor:
        """
        Evaluate the pre-trained model.

        Parameters
        ----------
        examples
            The raw classifier examples. They have a pair of a text
            and an aspect with the target sentiment.
        metric
        batch_size

        Returns
        -------

        """


@dataclass
class BertPipeline(Pipeline):
    model: BertABSClassifier
    tokenizer: transformers.BertTokenizer
    text_splitter: Callable[[str], List[str]] = None
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
        texts = self.text_splitter(text) if self.text_splitter else [text]
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

    def predict(self, input_batch: InputBatch) -> OutputBatch:
        # This implementation forces the model to return the detailed
        # output including hidden states and attentions.
        with tf.GradientTape() as tape:
            logits, hidden_states, attentions = self.model.call(
                token_ids=input_batch.token_ids,
                attention_mask=input_batch.attention_mask,
                token_type_ids=input_batch.token_type_ids
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
        #   hidden states: [input_batch, layer, sequence, embedding]
        #   attentions: [input_batch, layer, head, attention, attention]
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
            output_batch: OutputBatch
    ) -> Iterable[AspectSpanLabeled]:
        sentiment_ids = np.argmax(output_batch.scores, axis=-1).astype(int)
        for i, aspect_span in enumerate(aspect_spans):
            sentiment_id = sentiment_ids[i]
            aspect_representation, patterns = self.pattern_recognizer(
                aspect_span=aspect_span,
                hidden_states=output_batch.hidden_states[i],
                attentions=output_batch.attentions[i],
                attention_grads=output_batch.attention_grads[i]
            ) if self.pattern_recognizer else (None, None)
            kwargs = asdict(aspect_span)
            aspect_span_labeled = AspectSpanLabeled(
                sentiment=Sentiment(sentiment_id),
                scores=output_batch.scores[i].numpy().tolist(),
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
