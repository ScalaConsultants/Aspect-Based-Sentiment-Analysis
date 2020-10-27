import os
from unittest import mock

import pytest
import numpy as np
import tensorflow as tf
from testfixtures import LogCapture

from aspect_based_sentiment_analysis.training import (
    LossHistory,
    ModelCheckpoint,
    EarlyStopping,
    StopTraining
)


def test_model_checkpoint(monkeypatch):
    monkeypatch.setattr(os, 'makedirs', lambda x: x)
    monkeypatch.setattr(os, 'mkdir', lambda x: x)
    model = mock.Mock()
    model.save_pretrained = mock.MagicMock()
    history = mock.Mock()
    history.test = {1: 5, 2: 3, 3: 5}

    checkpoint = ModelCheckpoint(
        model, history, home_dir='', direction='minimize')

    with LogCapture() as log:
        for epoch in history.test:
            checkpoint.on_epoch_end(epoch)

    assert checkpoint.best_result == 3
    assert os.path.basename(checkpoint.best_model_dir) == 'epoch-02-3.00'
    assert all(record.name == 'absa.callbacks' for record in log.records)
    assert len(log.records) == 2

    history.test = {1: 3, 2: 4, 3: 5, 4: 5, 5: 5, 6: 5, 7: 5}
    checkpoint = ModelCheckpoint(
        model, history, home_dir='', direction='maximize')

    with LogCapture() as log:
        for epoch in history.test:
            checkpoint.on_epoch_end(epoch)
    assert checkpoint.best_result == 5
    assert checkpoint.best_model_dir == 'epoch-03-5.00'
    assert len(log.records) == 3


def test_early_stopping():
    history = mock.Mock()
    history.test = {1: 10, 2: 8, 3: 5, 4: 2, 5:2, 6:2, 7:2}
    callback = EarlyStopping(history, patience=1, min_delta=1, direction='minimize')
    with pytest.raises(StopTraining):
        loops = 0
        for epoch in history.test:
            callback.on_epoch_end(epoch)
            loops += 1
    assert loops == 4
    assert epoch == 5   # when stopped

    history.test = {1: 3, 2: 4, 3: 5, 4: 5, 5:5, 6:5, 7:5}
    callback = EarlyStopping(history, patience=3, min_delta=1, direction='maximize')
    with pytest.raises(StopTraining):
        loops = 0
        for epoch in history.test:
            callback.on_epoch_end(epoch)
            loops += 1
    assert loops == 5
    assert epoch == 6   # when stopped


def test_loss_history_callback():
    history = LossHistory(verbose=True)
    # The simplified routine.
    batch_input = None
    with LogCapture() as log:
        for epoch in np.arange(1, 10 + 1):
            history.on_epoch_begin(epoch)
            for batch in range(100):
                train_loss = np.random.normal(loc=11-epoch, scale=0.5, size=32)
                tf_train_loss = tf.convert_to_tensor(train_loss)
                train_step_outputs = [tf_train_loss, 'model_outputs']
                history.on_train_batch_end(batch, batch_input, *train_step_outputs)

                tf_test_loss = tf_train_loss + 1
                test_step_outputs = [tf_test_loss, 'model_outputs']
                history.on_test_batch_end(batch, batch_input, *test_step_outputs)
            history.on_epoch_end(epoch)

    # Check training / evaluation statistics
    epoch_train_loss = list(history.train.values())
    assert np.allclose(epoch_train_loss, np.arange(1, 11)[::-1], 0.1)
    epoch_test_loss = list(history.test.values())
    assert np.allclose(epoch_test_loss, np.arange(2, 12)[::-1], 0.1)

    # Check if details are correct. The training mean during
    # the last epoch should be equal 1.
    details = history.train_details[10]
    assert len(details) == 32 * 100
    assert round(np.array(details).mean()) == 1

    # Check log events
    assert len(log.records) == 2010
    assert log.records[0].name == 'absa.callbacks'
