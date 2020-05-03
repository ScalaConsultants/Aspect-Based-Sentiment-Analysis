import logging
from abc import ABC
from abc import abstractmethod
from collections import OrderedDict
from dataclasses import asdict
from dataclasses import dataclass
from typing import Callable
from typing import Iterable
from typing import List

import numpy as np
import tensorflow as tf
import transformers

from . import alignment
from . import utils
from .data_types import TokenizedExample, Example, LabeledExample
from .data_types import PredictedExample
from .data_types import SubTask
from .data_types import CompletedSubTask
from .data_types import Task
from .data_types import CompletedTask
from .data_types import InputBatch
from .data_types import OutputBatch
from .data_types import Sentiment
from .models import ABSClassifier
from .models import BertABSClassifier
from .probing import PatternRecognizer
from .training import classifier_loss

logger = logging.getLogger('absa.pipeline')


@dataclass
class Pipeline(ABC):
    """
    The pipeline simplifies the use of the fine-tuned Aspect-Based Sentiment
    Classifier. The aim is to classify the sentiment of a potentially long
    text for several aspects. Furthermore, the pipeline gives the reasons for
    a decision, so we can infer how much results are reliable. For the basic
    core, you benefit from the `__call__` method.

    We made two important design decisions. Firstly, even some research
    presents how to predict several aspects at once, we process aspects
    independently. The used model does not support the multi-aspect
    prediction. Secondly, the sentiment of a long text tends to be fuzzy and
    neutral. Therefore, we split a text into smaller independent chunks,
    called spans. They can include a single sentence or several sentences.
    It depends how works a `text_splitter`. Note that longer spans have
    richer context information, but they requires significant computation
    resources.

    Thanks to the `pattern_recognizer`, we can investigate how much
    predictions are reliable. In our task, we are curious about two things
    at most. Firstly, we want to be sure that the model connects the correct
    word or words with the aspect. If the model does it wrong, the sentiment
    concerns the different entities. Secondly, even if the model recognized
    the aspect correctly, we wish to understand the model reasoning better.
    To do so, the pattern recognizer discovers patterns, a weighted sequence
    of words, and their approximated impact to the prediction. We want to
    avoid a situation wherein a single word or weird word combination
    triggers the model.

    Please note that the package contains the separated submodule
    `absa.training`. You can find there complete routines to tune or train
    either the language model or the classifier. Check out the example on
    the package website.
    """
    model: ABSClassifier
    tokenizer: transformers.PreTrainedTokenizer
    text_splitter: Callable[[str], List[str]]
    pattern_recognizer: PatternRecognizer

    @abstractmethod
    def __call__(self, text: str, aspects: List[str]) -> CompletedTask:
        """
        The __call__ method is for the basic core. The pipeline
        performs several clear transformations:
            - convert text and aspects into the task:
              the form of well-prepared tokenized example,
            - encode example into the model compatible input batch,
            - pass input batch to the model, and form the output batch,
            - label example using the output batch,
            - return the well-structured completed task.

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
        Preprocess the raw task text and aspects into the task. The task
        keeps text and aspects in the form of well-prepared tokenized
        example. The example is an independent *preprocessed* sample,
        tokenized pair of two strings (text, aspect) which we further
        encode and pass to the model.

        Note that we may need to split a long text into smaller text chunks,
        called spans. We can do it using a text splitter which defines how
        long the span is.

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
    def tokenize(self, examples: List[Example]) -> List[TokenizedExample]:
        """
        Tokenize the example. The model can not process the raw pair of two
        strings (text, aspect) directly.

        Parameters
        ----------
        examples
            List of example, the pairs of two raw strings (text, aspect).

        Returns
        -------
        tokenized_examples
            Independent *preprocessed* tokenized example.
        """

    @abstractmethod
    def encode(self, examples: List[TokenizedExample]) -> InputBatch:
        """
        Encode tokenized example. The input batch is a container of tensors
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

    @abstractmethod
    def label(
            self,
            examples: List[TokenizedExample],
            output_batch: OutputBatch
    ) -> Iterable[PredictedExample]:
        """
        Label example using the detailed information about the prediction.
        The predicted example contains additional attributes such as the
        sentiment and scores for each sentiment class. The aspect product
        and patterns are optional. They are if a pipeline has a **pattern
        recognizer**.

        Parameters
        ----------
        examples
            Independent *preprocessed* tokenized example.
        output_batch
            Container of tensors describing a prediction.

        Returns
        -------
        predicted_examples
            List of labeled example.
        """

    @abstractmethod
    def evaluate(
            self,
            examples: List[LabeledExample],
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
class BertPipeline(Pipeline):
    model: BertABSClassifier
    tokenizer: transformers.BertTokenizer
    text_splitter: Callable[[str], List[str]] = None
    pattern_recognizer: PatternRecognizer = None

    def __call__(
            self,
            text: str,
            aspects: List[str]
    ) -> CompletedTask:
        task = self.preprocess(text, aspects)
        tokenized_examples = task.batch
        input_batch = self.encode(tokenized_examples)
        output_batch = self.predict(input_batch)
        predicted_examples = self.label(tokenized_examples, output_batch)
        completed_task = self.get_completed_task(task, predicted_examples)
        return completed_task

    def preprocess(self, text: str, aspects: List[str]) -> Task:
        texts = self.text_splitter(text) if self.text_splitter else [text]
        subtasks = OrderedDict()
        for aspect in aspects:
            examples = [Example(text, aspect) for text in texts]
            tokenized_examples = self.tokenize(examples)
            subtask = SubTask(text, aspect, tokenized_examples)
            subtasks[aspect] = subtask
        task = Task(text, aspects, subtasks)
        return task

    def tokenize(self, examples: List[Example]) -> List[TokenizedExample]:
        examples = [alignment.tokenize(self.tokenizer, e.text, e.aspect)
                    for e in examples]
        return examples

    def encode(self, examples: List[TokenizedExample]) -> InputBatch:
        token_pairs = [(e.text_tokens, e.aspect_tokens) for e in examples]
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

    def label(
            self,
            examples: List[TokenizedExample],
            output_batch: OutputBatch
    ) -> Iterable[PredictedExample]:
        sentiment_ids = np.argmax(output_batch.scores, axis=-1).astype(int)
        for i, example in enumerate(examples):
            sentiment_id = sentiment_ids[i]
            aspect_representation, patterns = self.pattern_recognizer(
                example=example,
                hidden_states=output_batch.hidden_states[i],
                attentions=output_batch.attentions[i],
                attention_grads=output_batch.attention_grads[i]
            ) if self.pattern_recognizer else (None, None)
            kwargs = asdict(example)
            predicted_example = PredictedExample(
                sentiment=Sentiment(sentiment_id),
                scores=output_batch.scores[i].numpy().tolist(),
                aspect_representation=aspect_representation,
                patterns=patterns,
                **kwargs
            )
            yield predicted_example

    def evaluate(
            self,
            examples: List[LabeledExample],
            metric: tf.metrics.Metric,
            batch_size: int
    ) -> tf.Tensor:
        batches = utils.batches(examples, batch_size)
        for batch in batches:
            tokenized_examples = self.tokenize(batch)
            input_batch = self.encode(tokenized_examples)
            output_batch = self.predict(input_batch)
            labeled_examples = self.label(tokenized_examples, output_batch)
            y_pred = [e.sentiment.value for e in labeled_examples]
            y_true = [e.sentiment.value for e in batch]
            metric.update_state(y_true, y_pred)
        result = metric.result()
        return result

    @staticmethod
    def get_completed_task(
            task: Task,
            batch_examples: Iterable[PredictedExample]
    ) -> CompletedTask:
        batch_examples = list(batch_examples)
        subtasks = OrderedDict()
        for start, end in task.indices:
            examples = batch_examples[start:end]
            first = examples[0]
            scores = np.max([e.scores for e in examples], axis=0)
            scores /= np.linalg.norm(scores, ord=1)
            sentiment_id = np.argmax(scores).astype(int)
            aspect_document = CompletedSubTask(
                text=task.text,
                aspect=first.aspect,
                examples=examples,
                sentiment=Sentiment(sentiment_id),
                scores=list(scores)
            )
            subtasks[first.aspect] = aspect_document
        task = CompletedTask(task.text, task.aspects, subtasks)
        return task
