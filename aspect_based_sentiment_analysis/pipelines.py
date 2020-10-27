import logging
from abc import ABC
from abc import abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable
from typing import Iterable
from typing import List

import numpy as np
import tensorflow as tf
import transformers

from . import alignment
from . import utils
from .data_types import TokenizedExample
from .data_types import Example
from .data_types import LabeledExample
from .data_types import PredictedExample
from .data_types import SubTask
from .data_types import CompletedSubTask
from .data_types import Task
from .data_types import CompletedTask
from .data_types import InputBatch
from .data_types import OutputBatch
from .data_types import Sentiment
from .models import BertABSClassifier
from .training import classifier_loss
from .professors import Professor

logger = logging.getLogger('absa.pipeline')


@dataclass
class _Pipeline(ABC):
    """
    The pipeline simplifies the use of the fine-tuned Aspect-Based Sentiment
    Classifier. The aim is to classify the sentiment of a potentially long
    text for several aspects. Furthermore, the pipeline gives the reasons for
    a decision, so we can infer how much results are reliable. For the basic
    absa, you benefit from the `__call__` method.

    Please note that the package contains the separated submodule
    `absa.training`. You can find there complete routines to tune or train
    either the language model or the classifier. Check out the example on
    the package website.
    """

    @abstractmethod
    def __call__(self, text: str, aspects: List[str]) -> CompletedTask:
        """
        The __call__ method is for the basic inference to make predictions.

        Parameters
        ----------
        text
            This is the raw task text without rigorous length limit.
            The text_splitter, if provided, splits the text into smaller spans,
            and the pipeline processes them independently.
        aspects
            The aspects for which the pipeline does the sentiment analysis.
            For now, only the single word should describe an aspect
            (one token, do not combine words using a hyphen).

        Returns
        -------
        completed_task
            The labeled example after the classification.
        """

    @abstractmethod
    def preprocess(self, text: str, aspects: List[str]) -> Task:
        """
        Preprocess the raw task text and aspects into the task. Note that
        we may need to split a long text into smaller text chunks, called
        spans. We can do it using a text splitter which defines how long
        the span is.

        Parameters
        ----------
        text
            This is the raw task text without rigorous length limit.
            The pipeline can contain the text_splitter, which splits a text
            into smaller chunks, called spans.
        aspects
            The aspects for which the pipeline does the sentiment analysis.

        Returns
        -------
        task
            Text and aspects in the form of well-prepared tokenized example.
        """

    @abstractmethod
    def tokenize(self, examples: Iterable[Example]) -> Iterable[
        TokenizedExample]:
        """
        Tokenize the example. The model can not process the raw pair of two
        strings (text, aspect) directly.

        Parameters
        ----------
        examples
            Iterable of examples, the pairs of two raw strings (text, aspect).

        Returns
        -------
        tokenized_examples
            Independent *preprocessed* tokenized example.
        """

    @abstractmethod
    def encode(self, examples: Iterable[TokenizedExample]) -> InputBatch:
        """
        Encode tokenized examples. The input batch is a container of tensors
        crucial for the model to make a prediction. The names are compatible 
        with the *transformers* package. 

        Parameters
        ----------
        examples
            Independent *preprocessed* tokenized example.

        Returns
        -------
        input_batch
            Container of tensors needed to make a prediction.
        """

    @abstractmethod
    def predict(self, input_batch: InputBatch) -> OutputBatch:
        """
        Pass the input batch to the pretrained model to make a prediction.
        The pipeline collects not only scores, the softmax of logits,
        but also hidden states, attentions, and attention gradients with
        respect to the model output. In the end, the method packs them into
        the output batch and returns.

        Parameters
        ----------
        input_batch
            Container of tensors needed to make a prediction.

        Returns
        -------
        output_batch
            Container of tensors describing a prediction.
        """

    @staticmethod
    def postprocess(
            task: Task,
            batch_examples: Iterable[PredictedExample]
    ) -> CompletedTask:
        """
        Postprocess using the detailed information about the prediction.
        The predicted examples contains additional attributes such as the
        sentiment and scores for each sentiment class.

        Parameters
        ----------
        task
            Text and aspects in the form of well-prepared tokenized example.
        batch_examples
            Predicted examples that come from a professor.

        Returns
        -------
        CompletedTask
            Return the completed task with predicted examples.
        """

    @abstractmethod
    def evaluate(
            self,
            examples: Iterable[LabeledExample],
            metric: tf.metrics.Metric,
            batch_size: int
    ) -> tf.Tensor:
        """
        Evaluate the pre-trained model.

        Parameters
        ----------
        examples
            Labeled true example.
        metric
            TensorFlow metric.
        batch_size
            Number of samples in a batch.

        Returns
        -------
        result
            Metric value tensor.
        """


@dataclass
class Pipeline(_Pipeline):
    model: BertABSClassifier
    tokenizer: transformers.BertTokenizer
    professor: Professor
    text_splitter: Callable[[str], List[str]] = None

    def __call__(self, text: str, aspects: List[str]) -> CompletedTask:
        task = self.preprocess(text, aspects)
        predictions = self.transform(task.examples)
        completed_task = self.postprocess(task, predictions)
        return completed_task

    def preprocess(self, text: str, aspects: List[str]) -> Task:
        spans = self.text_splitter(text) if self.text_splitter else [text]
        subtasks = OrderedDict()
        for aspect in aspects:
            examples = [Example(span, aspect) for span in spans]
            subtasks[aspect] = SubTask(text, aspect, examples)
        task = Task(text, aspects, subtasks)
        return task

    def transform(self, examples: Iterable[Example]) -> Iterable[PredictedExample]:
        tokenized_examples = self.tokenize(examples)
        input_batch = self.encode(tokenized_examples)
        output_batch = self.predict(input_batch)
        predictions = self.review(tokenized_examples, output_batch)
        return predictions

    def tokenize(self, examples: Iterable[Example]) -> List[TokenizedExample]:
        return [alignment.tokenize(self.tokenizer, e.text, e.aspect) for e in examples]

    def encode(self, examples: Iterable[TokenizedExample]) -> InputBatch:
        token_pairs = [(e.text_subtokens, e.aspect_subtokens) for e in examples]
        encoded = self.tokenizer.batch_encode_plus(
            token_pairs,
            add_special_tokens=True,
            padding=True,
            return_tensors='tf',
            return_attention_masks=True
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
            predictions = tf.argmax(logits, axis=-1)
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

    def review(
            self,
            examples: Iterable[TokenizedExample],
            output_batch: OutputBatch
    ) -> Iterable[PredictedExample]:
        return (self.professor.review(e, o) for e, o in zip(examples, output_batch))

    @staticmethod
    def postprocess(
            task: Task,
            batch_examples: Iterable[PredictedExample]
    ) -> CompletedTask:
        batch_examples = list(batch_examples)  # Materialize examples.
        subtasks = OrderedDict()
        for start, end in task.indices:
            examples = batch_examples[start:end]
            # Examples should have the same aspect (an implicit check).
            aspect, = {e.aspect for e in examples}
            scores = np.max([e.scores for e in examples], axis=0)
            scores /= np.linalg.norm(scores, ord=1)
            sentiment_id = np.argmax(scores).astype(int)
            aspect_document = CompletedSubTask(
                text=task.text,
                aspect=aspect,
                examples=examples,
                sentiment=Sentiment(sentiment_id),
                scores=list(scores)
            )
            subtasks[aspect] = aspect_document
        task = CompletedTask(task.text, task.aspects, subtasks)
        return task

    def evaluate(
            self,
            examples: Iterable[LabeledExample],
            metric: tf.metrics.Metric,
            batch_size: int
    ) -> tf.Tensor:
        batches = utils.batches(examples, batch_size)
        for batch in batches:
            predictions = self.transform(batch)
            y_pred = [e.sentiment.value for e in predictions]
            y_true = [e.sentiment.value for e in batch]
            metric.update_state(y_true, y_pred)
        result = metric.result()
        return result
