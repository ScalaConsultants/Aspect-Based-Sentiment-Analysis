from typing import Iterable
from typing import List

import tensorflow as tf
import transformers

from . import routines
from .callbacks import Callback
from .data_types import LanguageModelTrainBatch


def train_language_model(
        model: transformers.TFBertForPreTraining,
        optimizer: tf.keras.optimizers.Optimizer,
        train_dataset: Iterable[LanguageModelTrainBatch],
        epochs: int,
        test_dataset: Iterable[LanguageModelTrainBatch] = None,
        callbacks: List[Callback] = None,
        strategy: tf.distribute.Strategy = tf.distribute.OneDeviceStrategy('CPU')
):
    """ Post train (fine-tune) the pretrained language model. """
    with strategy.scope():

        def train_step(*batch: List[tf.Tensor]):
            token_ids, attention_mask, token_type_ids, *targets = batch
            with tf.GradientTape() as tape:
                model_outputs = model.call(token_ids,
                                           attention_mask=attention_mask,
                                           token_type_ids=token_type_ids,
                                           training=True)
                loss_value = language_model_loss(...)

            variables = model.trainable_variables
            grads = tape.gradient(loss_value, variables)
            optimizer.apply_gradients(zip(grads, variables))
            return [loss_value, *model_outputs]

        def test_step(*batch: List[tf.Tensor]):
            token_ids, attention_mask, token_type_ids, *targets = batch
            model_outputs = model.call(token_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)
            loss_value = language_model_loss(...)
            return [loss_value, *model_outputs]

    routines.train(
        strategy=strategy,
        train_step=train_step,
        train_dataset=train_dataset,
        test_step=test_step,
        test_dataset=test_dataset,
        epochs=epochs,
        callbacks=callbacks
    )


def language_model_loss(*args) -> tf.Tensor:
    """ """
    raise NotImplemented
