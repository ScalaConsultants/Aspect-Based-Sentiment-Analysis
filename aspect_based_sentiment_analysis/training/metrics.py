import tensorflow as tf


class ConfusionMatrix(tf.metrics.Metric):
    """ Collect partial classification results
    directly into the Confusion Matrix. """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.data = self.add_weight(name='confusion-matrix',
                                    shape=[num_classes, num_classes],
                                    initializer='zeros',
                                    dtype=tf.dtypes.int32)

    def update_state(self, y_true, y_pred):
        batch = tf.math.confusion_matrix(
            y_true, y_pred,
            num_classes=self.num_classes,
            dtype=tf.dtypes.int32
        )
        self.data.assign_add(batch)

    def result(self):
        return self.data
