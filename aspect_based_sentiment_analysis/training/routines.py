import dataclasses
import logging
from typing import Callable
from typing import Iterable
from typing import List

import numpy as np
import tensorflow as tf

from .callbacks import Callback
from .callbacks import CallbackList
from .data_types import TrainBatch
from .errors import StopTraining

logger = logging.getLogger('absa.routines')


def train(strategy: tf.distribute.Strategy,
          train_step: Callable,
          train_dataset: Iterable[TrainBatch],
          test_step: Callable = None,
          test_dataset: Iterable[TrainBatch] = None,
          epochs: int = 10,
          callbacks: List[Callback] = None):
    callbacks = CallbackList(callbacks if callbacks else [])
    try:
        for epoch in np.arange(1, epochs+1):
            callbacks.on_epoch_begin(epoch)
            train_loop(train_step, train_dataset, callbacks, strategy)
            if test_step and test_dataset:
                test_loop(test_step, test_dataset, callbacks, strategy)
            callbacks.on_epoch_end(epoch)
    except StopTraining:
        logger.info('The training routine is stopped.')


def train_loop(train_step: Callable,
               dataset: Iterable[TrainBatch],
               callbacks: Callback,
               strategy: tf.distribute.Strategy):
    step = wrap_step_into_strategy(train_step, strategy)
    for i, batch in enumerate(dataset):
        tf_batch = dataclasses.astuple(batch)
        train_step_outputs = step(tf_batch)
        callbacks.on_train_batch_end(i, batch, *train_step_outputs)


def test_loop(test_step: Callable,
              dataset: Iterable[TrainBatch],
              callbacks: Callback,
              strategy: tf.distribute.Strategy):
    step = wrap_step_into_strategy(test_step, strategy)
    for i, batch in enumerate(dataset):
        tf_batch = dataclasses.astuple(batch)
        test_step_outputs = step(tf_batch)
        callbacks.on_test_batch_end(i, batch, *test_step_outputs)


def wrap_step_into_strategy(step: Callable, strategy: tf.distribute.Strategy):
    """ """
    def one_device(batch):
        return strategy.run(step, args=batch)

    def distributed(batch):
        dataset = tf.data.Dataset.from_tensors(batch)
        dist_batch, = strategy.experimental_distribute_dataset(dataset)
        per_replica_outputs = strategy.run(step, args=dist_batch)
        with tf.device('CPU'):
            return [tf.concat(per_replica_output.values, axis=0)
                    for per_replica_output in per_replica_outputs]

    is_distributed = isinstance(strategy, tf.distribute.MirroredStrategy)
    return distributed if is_distributed else one_device
