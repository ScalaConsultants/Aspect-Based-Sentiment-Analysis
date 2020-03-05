from typing import Callable
from typing import Iterable
from typing import List

import numpy as np
import tensorflow as tf
import transformers

from . import losses
from .callbacks import Callback
from .callbacks import CallbackList
from .models import BertABSClassifier
from .preprocessing.language_model_input import LanguageModelTrainBatch
from .preprocessing.extractor_model_input import ExtractorTrainBatch
from .preprocessing.classifier_model_input import ClassifierTrainBatch


def train(train_step: Callable,
          train_dataset: Iterable,
          test_step: Callable = None,
          test_dataset: Iterable = None,
          epochs: int = 10,
          callbacks: List = None):
    callbacks = CallbackList(callbacks if callbacks else [])
    for epoch in np.arange(1, epochs+1):
        callbacks.on_epoch_begin(epoch)
        train_loop(train_step, train_dataset, callbacks)
        if test_step and test_dataset:
            test_loop(test_step, test_dataset, callbacks)
        callbacks.on_epoch_end(epoch)


def train_loop(train_step: Callable, dataset: Iterable, callbacks: Callback):
    for i, batch in enumerate(dataset):
        train_step_outputs = train_step(batch)
        callbacks.on_train_batch_end(i, batch, *train_step_outputs)
    callbacks.on_train_end()


def test_loop(test_step: Callable, dataset: Iterable, callbacks: Callback):
    for i, batch in enumerate(dataset):
        test_step_outputs = test_step(batch)
        callbacks.on_test_batch_end(i, batch, *test_step_outputs)
    callbacks.on_test_end()


def post_train(model: transformers.TFBertForPreTraining,
               optimizer: tf.keras.optimizers.Optimizer,
               train_dataset: Iterable[LanguageModelTrainBatch],
               epochs: int,
               test_dataset: Iterable[LanguageModelTrainBatch] = None,
               callbacks: List[Callback] = None):
    """ Post train (fine-tune) the pretrained language model. """
    def train_step(batch: LanguageModelTrainBatch):
        with tf.GradientTape() as tape:
            model_outputs = model.call(batch.token_ids,
                                       attention_mask=batch.attention_mask,
                                       token_type_ids=batch.token_type_ids,
                                       training=True)
            loss_value = losses.language_model_loss(...)

        variables = model.trainable_variables
        grads = tape.gradient(loss_value, variables)
        optimizer.apply_gradients(zip(grads, variables))
        return [loss_value, *model_outputs]

    def test_step(batch: LanguageModelTrainBatch):
        model_outputs = model.call(batch.token_ids,
                                   attention_mask=batch.attention_mask,
                                   token_type_ids=batch.token_type_ids)
        loss_value = losses.language_model_loss(...)
        return [loss_value, *model_outputs]

    train(train_step, train_dataset, test_step, test_dataset, epochs, callbacks)


def tune_extractor(model: BertABSClassifier,
                   optimizer: tf.keras.optimizers.Optimizer,
                   train_dataset: Iterable[ExtractorTrainBatch],
                   epochs: int,
                   test_dataset: Iterable[ExtractorTrainBatch] = None,
                   callbacks: List[Callback] = None):
    """ This routines tune the extractor along with the language model. """
    def train_step(batch: ExtractorTrainBatch):
        with tf.GradientTape() as tape:
            model_outputs = model.call_extractor(batch.token_ids,
                                                 attention_mask=batch.attention_mask,
                                                 training=True)
            logits, *details = model_outputs
            loss_value = losses.extractor_loss(...)

        variables = model.language_model.bert.trainable_variables \
                    + model.extractor.trainable_variables
        grads = tape.gradient(loss_value, variables)
        optimizer.apply_gradients(zip(grads, variables))
        return [loss_value, *model_outputs]

    def test_step(batch: ExtractorTrainBatch):
        model_outputs = model.call_extractor(batch.token_ids,
                                             attention_mask=batch.attention_mask)
        loss_value = losses.extractor_loss(...)
        return [loss_value, *model_outputs]

    train(train_step, train_dataset, test_step, test_dataset, epochs, callbacks)


def tune_classifier(model: BertABSClassifier,
                    optimizer: tf.keras.optimizers.Optimizer,
                    train_dataset: Iterable[ClassifierTrainBatch],
                    epochs: int,
                    test_dataset: Iterable[ClassifierTrainBatch] = None,
                    callbacks: List[Callback] = None):
    """ This routines tune the classifier along with the language model. """
    def train_step(batch: ClassifierTrainBatch):
        with tf.GradientTape() as tape:
            model_outputs = model.call_classifier(batch.token_ids,
                                                  attention_mask=batch.attention_mask,
                                                  token_type_ids=batch.token_type_ids,
                                                  training=True)
            logits, *details = model_outputs
            loss_value = losses.classifier_loss(batch.target_labels, logits)

        variables = model.language_model.bert.trainable_variables \
                    + model.classifier.trainable_variables
        grads = tape.gradient(loss_value, variables)
        optimizer.apply_gradients(zip(grads, variables))
        return [loss_value, *model_outputs]

    def test_step(batch: ClassifierTrainBatch):
        model_outputs = model.call_classifier(batch.token_ids,
                                              attention_mask=batch.attention_mask,
                                              token_type_ids=batch.token_type_ids)
        logits, *details = model_outputs
        loss_value = losses.classifier_loss(batch.target_labels, logits)
        return [loss_value, *model_outputs]

    train(train_step, train_dataset, test_step, test_dataset, epochs, callbacks)
