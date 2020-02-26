import logging
from abc import ABC
from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import List
import tensorflow as tf
logger = logging.getLogger('absa.callbacks')


class Callback(ABC):

    def on_epoch_begin(self, i: int):
        """ """

    def on_epoch_end(self, i: int):
        """ """

    def on_train_batch_end(self, i: int, batch, *train_step_outputs):
        """ """

    def on_train_end(self):
        """ """

    def on_test_batch_end(self, i: int, batch, *test_step_outputs):
        """ """

    def on_test_end(self):
        """ """


@dataclass
class CallbackList(Callback):
    callbacks: List[Callback]

    def on_epoch_begin(self, i: int):
        for callback in self.callbacks:
            callback.on_epoch_begin(i)

    def on_epoch_end(self, i: int):
        for callback in self.callbacks:
            callback.on_epoch_end(i)

    def on_train_batch_end(self, *args):
        for callback in self.callbacks:
            callback.on_train_batch_end(*args)

    def on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end()

    def on_test_batch_end(self, *args):
        for callback in self.callbacks:
            callback.on_test_batch_end(*args)

    def on_test_end(self):
        for callback in self.callbacks:
            callback.on_test_end()


@dataclass
class Logger(Callback):
    level: int = 20
    file_path: str = None
    msg_format: str = '%(asctime)s [%(levelname)-6s] [%(name)-10s] %(message)s'

    def __post_init__(self):
        logger.setLevel(self.level)
        logger.propagate = False
        formatter = logging.Formatter(self.msg_format, datefmt='%Y-%m-%d %H:%M:%S')
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        # Handle all messages from the logger (not set the handler level)
        logger.addHandler(console)
        if self.file_path:
            file_handler = logging.FileHandler(self.file_path, mode='w')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)


@dataclass
class History(Callback):
    epoch: int = 0
    train_metric: tf.keras.metrics.Metric = field(default_factory=tf.keras.metrics.Mean)
    test_metric: tf.keras.metrics.Metric = field(default_factory=tf.keras.metrics.Mean)
    train: Dict = field(default_factory=dict)
    test: Dict = field(default_factory=dict)
    train_details: Dict = field(default_factory=dict)
    test_details: Dict = field(default_factory=dict)

    def on_epoch_begin(self, i: int):
        """ Resets all of the metric state variables. """
        self.epoch = i+1
        self.train_details[self.epoch] = []
        self.test_details[self.epoch] = []
        self.train_metric.reset_states()
        self.test_metric.reset_states()

    def on_epoch_end(self, i: int):
        self.train[self.epoch] = self.train_metric.result().numpy()
        self.test[self.epoch] = self.test_metric.result().numpy()
        message = f'Epoch {self.epoch:3d}    ' \
                  f'Average Train Loss: {self.train[self.epoch]:.5f}    ' \
                  f'Average Test Loss: {self.test[self.epoch]:.5f}'
        logger.info(message)

    def on_train_batch_end(self, i: int, batch, *train_step_outputs):
        loss_value, *model_outputs = train_step_outputs
        self.train_metric(loss_value)
        self.train_details[self.epoch].extend(loss_value.numpy())

    def on_test_batch_end(self, i: int, batch, *test_step_outputs):
        loss_value, *model_outputs = test_step_outputs
        self.test_metric(loss_value)
        self.test_details[self.epoch].extend(loss_value.numpy())
