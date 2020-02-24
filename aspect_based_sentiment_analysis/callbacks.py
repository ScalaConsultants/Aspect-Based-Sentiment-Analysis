import logging
from abc import ABC
from dataclasses import dataclass
from typing import List
logger = logging.getLogger('callbacks')


class Callback(ABC):

    def on_epoch_begin(self, i: int):
        """ """

    def on_epoch_end(self, i: int):
        """ """

    def on_train_batch_end(self, i: int, batch, *model_outputs):
        """ """

    def on_train_end(self):
        """ """

    def on_test_batch_end(self, i: int, batch, *model_outputs):
        """ """

    def on_test_end(self):
        """ """


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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
