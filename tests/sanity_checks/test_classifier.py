import os
import numpy as np
import tensorflow as tf
import transformers
import pytest
import aspect_based_sentiment_analysis as absa
np.random.seed(1)
tf.random.set_seed(1)


@pytest.mark.sanity_check
def test_sanity_classifier():
    # This sanity test verifies and presents how the classifier works. To
    # build our model, we have to define a config, which contains all
    # required information to build the `BertABSClassifier` model (including
    # the BERT language model). In this example, we use default parameters (
    # which are set up for our best performance), but of course, you can pass
    # your own parameters (maybe you would be interested to change the number
    # of polarities to classify, or properties of the BERT).
    base_model_name = 'bert-base-uncased'
    config = absa.BertABSCConfig.from_pretrained(base_model_name)
    model = absa.BertABSClassifier.from_pretrained(base_model_name, config=config)
    tokenizer = transformers.BertTokenizer.from_pretrained(base_model_name)

    # The first step to train the model is to define a dataset. The dataset
    # can be understood as a non-differential part of the pipeline for the
    # training. The dataset knows how to transform human-understandable
    # examples into model understandable batches. You are not obligated to
    # use datasets, you can create your own iterable, which transforms
    # classifier examples to classifier train batches.
    example = absa.ClassifierExample(
        text='The breakfast was delicious, really great.',
        aspect=absa.Aspect(name='breakfast', label=absa.Label.positive)
    )
    dataset = absa.ClassifierDataset(examples=[example, example],
                                     tokenizer=tokenizer,
                                     batch_size=2)

    # To easily adjust optimization process to our needs, we define custom
    # training loops called routines (in contrast to use built-in methods as
    # `fit`). Each routine has its own optimization step wherein we can
    # control which and how parameters are updated (according to the custom
    # training paradigm presented in TensorFlow 2.0). We iterate over a
    # dataset, perform train/test optimization steps, and collect results
    # using callbacks (which have a similar interface as tf.keras.Callback).
    # Please take a look at the `tune_classifier` function for more details.
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-8)
    logger, history = absa.Logger(), absa.History()
    absa.tune_classifier(model, optimizer, dataset, epochs=10, callbacks=[logger, history])

    # Our model should easily overfit, even in 10 iterations.
    assert np.isclose(history.train[1], 1, atol=0.1)
    assert history.train[10] < 1e-2

    # At the end, we would like to save the model. Our implementation
    # gentle extend the *transformers* lib capabilities, in consequences,
    # `BertABSClassifier` inherits from the `TFBertPreTrainedModel`, and
    # we can do a serialization easily.
    model.save_pretrained('.')

    # To make sure that the model serving works fine, we initialize the model
    # and the config once again. We perform the check on a single batch.
    del model, config
    config = absa.BertABSCConfig.from_pretrained('.')
    model = absa.BertABSClassifier.from_pretrained('.', config=config)
    batch = next(iter(dataset))
    model_outputs = model.call_classifier(batch.token_ids,
                                          attention_mask=batch.attention_mask,
                                          token_type_ids=batch.token_type_ids)
    logits, *details = model_outputs
    loss_value = absa.losses.classifier_loss(batch.target_labels, logits)
    train_loss = loss_value.numpy().mean()
    assert train_loss < 1e-2

    # The training procedure is roughly verified. Now, using our tuned model,
    # we can build the `BertPipeline`. The pipeline is the high level
    # interface to perform predictions. The model should be highly confident
    # that this is the positive example (verify the softmax scores).
    nlp = absa.BertPipeline(model, tokenizer)
    aspect_prediction, = nlp.predict(example.text, aspect_names=['breakfast'])
    assert aspect_prediction.label == absa.Label.positive
    assert np.allclose(aspect_prediction.scores, [0.0, 0.0, 0.99], atol=0.01)

    # That's all, clean up the configuration, and the temporary saved model.
    os.remove('config.json')
    os.remove('tf_model.h5')
