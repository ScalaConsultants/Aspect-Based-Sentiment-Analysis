from unittest import mock
from dataclasses import asdict

import numpy as np
import pytest
import tensorflow as tf

import aspect_based_sentiment_analysis as absa
from aspect_based_sentiment_analysis import utils


@pytest.fixture
@utils.cache_fixture
def inputs(request):  # The cache function uses the `request` parameter.
    nlp = absa.load('absa/classifier-rest-0.1',
                    output_attentions=True,
                    output_hidden_states=True)
    text = ("We are great fans of Slack, but we wish the subscriptions "
            "were more accessible to small startups.")
    aspect = 'slack'
    document_batch = nlp.preprocess(pairs=[(text, aspect)])
    document, *_ = document_batch.documents
    model_outputs = nlp.predict(document_batch)
    model_outputs = [tensor[0] for tensor in model_outputs]

    # Covert Document and EagerTensor's to the native python objects
    # and facilitate the serialization process.
    raw_document = asdict(document)
    raw_model_outputs = [tensor.numpy().tolist() for tensor in model_outputs]
    return raw_document, raw_model_outputs


@pytest.mark.skip
def test_integration(inputs):
    document_kwargs, model_outputs = inputs
    document = absa.Document(**document_kwargs)
    model_outputs = [tf.convert_to_tensor(o) for o in model_outputs]
    scores, *details = model_outputs

    recognizer = absa.AttentionPatternRecognizer()
    aspect_pattern, patterns = recognizer(document, *details)

    assert patterns


@pytest.mark.skip
def test_construct_patterns():
    recognizer = absa.AttentionPatternRecognizer
    document = mock.MagicMock()
    impacts = [1, 2, 3]
    mixtures = np.arange(12).reshape(3, 4).tolist()
    patterns = recognizer.construct_patterns(document, impacts, mixtures)
    assert len(patterns) == 3
    pattern_1, pattern_2, pattern_3 = patterns
    assert pattern_2.impact == 2
    assert pattern_2.weights == [4, 5, 6, 7]


@pytest.mark.skip
def test_get_key_mixtures():
    recognizer = absa.AttentionPatternRecognizer
    impacts = np.array([1, 2, 3])
    mixtures = np.arange(12).reshape(3, 4)

    key_impacts, key_mixtures = recognizer.get_key_mixtures(
        impacts, mixtures, percentile=80)
    assert len(key_impacts) == len(key_mixtures) == 2
    assert key_impacts == (3, 2)
    mixtures_1, mixtures_2 = key_mixtures
    assert mixtures_1 == [8, 9, 10, 11]
    assert mixtures_2 == [4, 5, 6, 7]

    key_impacts, key_mixtures = recognizer.get_key_mixtures(
        impacts, mixtures, percentile=10)
    assert len(key_impacts) == 1
    assert key_impacts == (3,)


@pytest.mark.skip
def test_mask_noise():
    recognizer = absa.AttentionPatternRecognizer
    interest = np.array([[1, -1, 5, 6],
                         [2, -1, 2, 1],
                         [0, -1, -5, 1]])

    clean_interest = recognizer.mask_noise(interest, percentile=70)
    assert clean_interest.tolist() == [[0, 0, 5, 6],
                                       [2, 0, 2, 0],
                                       [0, 0, -5, 0]]
    magnitude = lambda x: np.sum(np.abs(x))
    ratio = magnitude(clean_interest) / magnitude(interest)
    assert round(ratio, 2) == 0.77

    clean_interest = recognizer.mask_noise(interest, percentile=20)
    ratio = magnitude(clean_interest) / magnitude(interest)
    assert round(ratio, 2) == 0.23
