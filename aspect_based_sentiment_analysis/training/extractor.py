from typing import Iterable
from typing import List

import tensorflow as tf

from .. import routines
from ..callbacks import Callback
from ..models import BertABSClassifier
from ..preprocessing.extractor_model import ExtractorTrainBatch


def train_extractor(
        model: BertABSClassifier,
        optimizer: tf.keras.optimizers.Optimizer,
        train_dataset: Iterable[ExtractorTrainBatch],
        epochs: int,
        test_dataset: Iterable[ExtractorTrainBatch] = None,
        callbacks: List[Callback] = None,
        strategy: tf.distribute.Strategy = tf.distribute.OneDeviceStrategy('CPU')
):
    """ This routines tune the extractor along with the language model. """
    with strategy.scope():

        def extractor_loss(*args) -> tf.Tensor:
            """ """
            raise NotImplemented

        def train_step(*batch: List[tf.Tensor]):
            token_ids, attention_mask, target_sequence_labels = batch
            with tf.GradientTape() as tape:
                model_outputs = model.call_extractor(token_ids,
                                                     attention_mask=attention_mask,
                                                     training=True)
                logits, *details = model_outputs
                loss_value = extractor_loss(...)

            variables = model.language_model.bert.trainable_variables \
                        + model.extractor.trainable_variables
            grads = tape.gradient(loss_value, variables)
            optimizer.apply_gradients(zip(grads, variables))
            return [loss_value, *model_outputs]

        def test_step(*batch: List[tf.Tensor]):
            token_ids, attention_mask, target_sequence_labels = batch
            model_outputs = model.call_extractor(token_ids,
                                                 attention_mask=attention_mask)
            loss_value = extractor_loss(...)
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

