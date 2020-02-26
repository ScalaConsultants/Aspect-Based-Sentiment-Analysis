import tensorflow as tf


def language_model_loss(*args):
    """ """


def extractor_loss(*args):
    """ """


def classifier_loss(labels, logits):
    """ The classifier aim is to predict the three classes: positive,
    negative and neutral with respect to an aspect-target. In such case,
    we use the cross entropy loss.

    Note that we can also try to formulate this problem as a binary
    classification, and use for example the Hinge Loss. In consequences,
    we would reduce the number of parameters 3 times. """
    return tf.nn.softmax_cross_entropy_with_logits(labels, logits, axis=-1, name='Loss')
