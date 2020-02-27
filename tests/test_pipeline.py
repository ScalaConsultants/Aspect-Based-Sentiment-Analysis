import pytest
import numpy as np
import transformers
import tensorflow as tf

from aspect_based_sentiment_analysis import BertABSCConfig
from aspect_based_sentiment_analysis import BertABSClassifier
from aspect_based_sentiment_analysis import Label
from aspect_based_sentiment_analysis import BertPipeline
np.random.seed(1)
tf.random.set_seed(1)


@pytest.fixture
def pipeline() -> BertPipeline:
    base_model_name = 'bert-base-uncased'
    config = BertABSCConfig.from_pretrained(base_model_name)
    model = BertABSClassifier.from_pretrained(base_model_name, config=config)
    tokenizer = transformers.BertTokenizer.from_pretrained(base_model_name)
    return BertPipeline(model, tokenizer)


def test_predict(pipeline):
    predictions = pipeline.predict(
        text='The breakfast was delicious, really great.',
        aspect_names=['breakfast', 'hotel']
    )
    prediction_1, prediction_2 = predictions
    # The classifier layer is initialized using the kind of a normal
    # distribution (TruncatedNormal), so we expect that each class should be
    # rather equal likely. Scores for different aspects should be different.
    assert np.allclose(prediction_1.scores, [0.33, 0.33, 0.33], atol=0.1)
    assert not np.allclose(prediction_1.scores, prediction_2.scores)


def test_build_aspect_predictions():
    text = 'The breakfast was delicious, really great.'
    aspect_name = 'breakfast'
    logits = tf.convert_to_tensor([[4, -2, 2],
                                   [1, -3, 2]], dtype=tf.float32)
    predictions = BertPipeline.build_aspect_predictions(
        [aspect_name, aspect_name], logits, text
    )
    prediction_1, prediction_2 = predictions
    assert prediction_1.label == Label.neutral
    assert prediction_2.label == Label.positive
    assert prediction_1.text == prediction_2.text == text
    # Each aspect prediction contains softmax scores for the validation purpose.
    assert np.allclose(prediction_1.scores, [0.87, 0.0, 0.11], atol=0.01)
