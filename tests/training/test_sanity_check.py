import os
import numpy as np
import tensorflow as tf
import transformers
import pytest

import aspect_based_sentiment_analysis as absa
from aspect_based_sentiment_analysis import (
    BertABSCConfig,
    BertABSClassifier,
    Pipeline,
    Professor,
    LabeledExample,
    Sentiment
)
from aspect_based_sentiment_analysis.training import (
    Logger,
    LossHistory
)


@pytest.mark.slow
def test_sanity_classifier():
    np.random.seed(1)
    tf.random.set_seed(1)
    # This sanity test verifies and presents how train a classifier. To
    # build our model, we have to define a config, which contains all required
    # information needed to build the `BertABSClassifier` model (including
    # the BERT language model). In this example, we use default parameters
    # (which are set up for our best performance), but of course, you can pass
    # your own parameters (maybe you would be interested to change the number
    # of polarities to classify, or properties of the BERT itself).
    base_model_name = 'bert-base-uncased'
    strategy = tf.distribute.OneDeviceStrategy('CPU')
    with strategy.scope():
        config = BertABSCConfig.from_pretrained(base_model_name)
        model = BertABSClassifier.from_pretrained(base_model_name, config=config)
        tokenizer = transformers.BertTokenizer.from_pretrained(base_model_name)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # The first step to train the model is to define a dataset. The dataset
    # can be understood as a non-differential part of the training pipeline
    # The dataset knows how to transform human-understandable example into
    # model understandable batches. You are not obligated to use datasets,
    # you can create your own iterable, which transforms classifier example
    # to the classifier train batches.
    example = LabeledExample(
        text='The breakfast was delicious, really great.',
        aspect='breakfast',
        sentiment=Sentiment.positive)
    dataset = absa.training.ClassifierDataset(
        examples=[example, example],
        tokenizer=tokenizer,
        batch_size=2)

    # To easily adjust optimization process to our needs, we define custom
    # training loops called routines (in contrast to use built-in methods as
    # the `fit`). Each routine has its own optimization step wherein we can
    # control which and how parameters are updated (according to the custom
    # training paradigm presented in the TensorFlow 2.0). We iterate over a
    # dataset, perform train/test optimization steps, and collect results
    # using callbacks (which have a similar interface as the tf.keras.Callback).
    # Please take a look at the `train_classifier` function for more details.
    logger, loss_value = Logger(), LossHistory()
    absa.training.train_classifier(
        model, optimizer, dataset,
        epochs=10,
        callbacks=[logger, loss_value],
        strategy=strategy)

    # Our model should easily overfit in just 10 iterations.
    assert .1 < loss_value.train[1] < 1
    assert loss_value.train[10] < 1e-4

    # In the end, we would like to save the model. Our implementation
    # gentle extend the *transformers* lib capabilities, in consequences,
    # `BertABSClassifier` inherits from the `TFBertPreTrainedModel`, and
    # we can do a serialization easily.
    model.save_pretrained('.')

    # To make sure that the model serving works fine, we initialize the model
    # and the config once again. We perform the check on a single example.
    del model, config
    config = BertABSCConfig.from_pretrained('.')
    model = BertABSClassifier.from_pretrained('.', config=config)
    batch = next(iter(dataset))
    model_outputs = model.call(
        batch.token_ids,
        attention_mask=batch.attention_mask,
        token_type_ids=batch.token_type_ids
    )
    logits, *details = model_outputs
    loss_fn = tf.nn.softmax_cross_entropy_with_logits
    loss_value = loss_fn(batch.target_labels, logits, axis=-1, name='Loss')
    loss_value = loss_value.numpy().mean()
    assert loss_value < 1e-4

    # The training procedure is roughly verified. Now, using our tuned model,
    # we can build the `BertPipeline`. The pipeline is the high level interface
    # to perform predictions. The model should be highly confident that this is
    # the positive example (verify the softmax scores).
    professor = Professor()
    nlp = Pipeline(model, tokenizer, professor)
    [breakfast] = nlp(example.text, aspects=['breakfast'])
    assert breakfast.sentiment == Sentiment.positive
    assert np.allclose(breakfast.scores, [0.0, 0.0, 0.99], atol=0.01)

    # That's all, clean up the configuration, and the temporary saved model.
    os.remove('config.json')
    os.remove('tf_model.h5')
