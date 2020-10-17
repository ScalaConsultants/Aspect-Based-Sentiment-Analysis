import os
from unittest.mock import MagicMock

import numpy as np
import tensorflow as tf
from testfixtures import LogCapture

from aspect_based_sentiment_analysis import (
    BertABSCConfig,
    BertABSClassifier
)
from aspect_based_sentiment_analysis.training import (
    LossHistory,
    ModelCheckpoint
)


def test_loss_history_callback():
    history = LossHistory()
    # The simplified routine.
    batch_input = None
    with LogCapture() as log:
        for epoch in np.arange(1, 10 + 1):
            history.on_epoch_begin(epoch)
            for batch in range(100):
                train_loss = np.random.normal(loc=11 - epoch, scale=0.5,
                                              size=32)
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
    assert len(log.records) == 10
    assert log.records[0].name == 'absa.callbacks'


def test_model_checkpoint(tmp_path):
    base_model_name = 'bert-base-uncased'
    config = BertABSCConfig.from_pretrained(base_model_name)
    model = BertABSClassifier.from_pretrained(base_model_name, config=config)

    loss_history = MagicMock()
    loss_history.test = {1: 5, 2: 3, 3: 5}
    checkpoint = ModelCheckpoint(model, loss_history, tmp_path)

    with LogCapture() as log:
        for epoch in np.arange(1, 3+1):
            checkpoint.on_epoch_begin(epoch)
            checkpoint.on_epoch_end(epoch)

    assert checkpoint.best_result == 3
    assert os.path.basename(checkpoint.best_model_dir) == 'epoch-02-3.00'
    records = [r for r in log.records if r.name == 'absa.callbacks']
    assert len(records) == 2
