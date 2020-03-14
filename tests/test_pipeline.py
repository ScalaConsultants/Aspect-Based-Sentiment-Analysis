import pytest
import numpy as np
import transformers
import tensorflow as tf

from aspect_based_sentiment_analysis import BertABSCConfig
from aspect_based_sentiment_analysis import BertABSClassifier
from aspect_based_sentiment_analysis import Sentiment
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


def test_call(pipeline):
    predictions = pipeline(
        text='The breakfast was delicious, really great.',
        aspects=['breakfast', 'hotel']
    )
    breakfast, hotel = predictions
    # The classifier layer is initialized using the kind of a normal
    # distribution (TruncatedNormal), so we expect that each class should be
    # rather equal likely. Scores for different aspects should be different.
    assert np.allclose(breakfast.scores, [0.33, 0.33, 0.33], atol=0.1)
    assert not np.allclose(breakfast.scores, hotel.scores)


def test_build_aspect_predictions():
    text = 'The breakfast was delicious, really great.'
    aspect = 'breakfast'
    # Each aspect prediction contains softmax scores for the validation purpose.
    scores = np.array([0.0, 0.11, 0.87])
    breakfast = BertPipeline.build_prediction(
        (text, aspect), scores
    )
    assert breakfast.aspect == aspect
    assert breakfast.sentiment == Sentiment.positive
    assert breakfast.text == text
    assert np.allclose(breakfast.scores, scores)
