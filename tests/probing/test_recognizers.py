from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

import aspect_based_sentiment_analysis as absa
from aspect_based_sentiment_analysis import alignment
from aspect_based_sentiment_analysis import Example
from aspect_based_sentiment_analysis.probing import AttentionGradientProduct


@pytest.fixture
def inputs():
    nlp = absa.load('absa/classifier-rest-0.1')
    text = ("We are great fans of Slack, but we wish the subscriptions "
            "were more accessible to small startups.")
    aspect = 'slack'
    example = Example(text, aspect)
    tokenized_examples = nlp.tokenize(examples=[example])
    input_batch = nlp.encode(tokenized_examples)
    output_batch = nlp.predict(input_batch)

    tokenized_example = tokenized_examples[0]
    attentions = alignment.merge_input_attentions(
        output_batch.attentions[0],
        alignment=tokenized_example.alignment
    )
    attention_grads = alignment.merge_input_attentions(
        output_batch.attention_grads[0],
        alignment=tokenized_example.alignment
    )
    return tokenized_example, attentions, attention_grads


def test_integration(inputs):
    tokenized_example, attentions, attention_grads = inputs
    recognizer = AttentionGradientProduct()
    aspect_repr, patterns = recognizer(
        example=tokenized_example,
        attentions=attentions,
        attention_grads=attention_grads,
        hidden_states=None  # It's unnecessary for this pattern recognizer.
    )

    index = np.argmax(np.abs(aspect_repr.look_at))
    assert aspect_repr.tokens[index] == 'slack'
    tokens = np.array(aspect_repr.tokens)
    look_at = np.array(aspect_repr.look_at)
    most_important = tokens[look_at > 0.2].tolist()
    assert most_important == ['fans', 'slack', '.']

    come_from = np.array(aspect_repr.come_from)
    most_important = tokens[come_from > 0.7].tolist()
    assert most_important == ['fans', 'slack', 'startups']

    assert len(patterns) == 8
    pattern_1, *_ = patterns
    assert np.isclose(pattern_1.impact, 1)
    weights = np.round(pattern_1.weights, decimals=2).tolist()
    assert weights[:6] == [0.12, 0.06, 0.24, 0.78, 0.09, 1.0]


def test_get_product():
    recognizer = AttentionGradientProduct()
    tf.random.set_seed(1)
    attentions = tf.random.normal([10, 10, 3, 3])
    attention_grads = tf.random.normal([10, 10, 3, 3])
    # Calculate partial results here by the hand.
    raw_product = tf.reduce_sum(attentions * attention_grads, axis=(0, 1))
    raw_product = np.round(raw_product.numpy().tolist(), decimals=2).tolist()

    product = recognizer.get_product(attentions, attention_grads)
    product = np.round(product.tolist(), decimals=2).tolist()
    assert product == raw_product


def test_get_patterns():
    def get_ratio(patterns):
        information = np.sum([p.weights for p in patterns])
        weights = product[[1, 2, 3, 4], :][:, [1, 2, 3, 4]]
        weights = weights / weights.max()
        return information / np.sum(weights)

    recognizer = AttentionGradientProduct(information_in_patterns=50)
    example = mock.MagicMock()
    example.tokens = ['CLS', 'this', 'is', 'a', 'test', 'SEP', 'test', 'SEP']
    example.text_tokens = ['this', 'is', 'a', 'test']
    product = np.arange(64).reshape([8, 8])

    patterns = recognizer.get_patterns(example, product)
    pattern_1, pattern_2 = patterns
    assert pattern_1.tokens == pattern_2.tokens == ['this', 'is', 'a', 'test']
    assert np.abs(pattern_1.impact) == 1
    assert np.allclose(pattern_1.weights, [0.917, 0.944, 0.972, 1.0], atol=0.01)
    assert np.abs(pattern_2.impact) == 0.75
    assert np.allclose(pattern_2.weights, [0.694, 0.722, 0.75, 0.778],
                       atol=0.01)
    assert get_ratio(patterns) > 0.5

    recognizer = AttentionGradientProduct(information_in_patterns=80)
    patterns = recognizer.get_patterns(example, product)
    assert len(patterns) == 3
    assert 0.9 > get_ratio(patterns) > 0.8


def test_get_aspect_representation():
    recognizer = AttentionGradientProduct()
    aspect_repr = mock.MagicMock()
    aspect_repr.tokens = ['CLS', 'this', 'is', 'a', 'test', 'SEP', 'test',
                          'SEP']
    aspect_repr.text_tokens = ['this', 'is', 'a', 'test']
    product = np.arange(64).reshape([8, 8])

    aspect_pattern = recognizer.get_aspect_representation(aspect_repr, product)
    assert aspect_pattern.tokens == ['this', 'is', 'a', 'test']
    assert np.allclose(aspect_pattern.come_from, [0.942, 0.962, 0.981, 1.0],
                       atol=0.01)
    assert np.allclose(aspect_pattern.look_at, [0.368, 0.579, 0.789, 1.0],
                       atol=0.01)


def test_input_validation():
    recognizer = AttentionGradientProduct  # Static method
    example = mock.MagicMock()
    example.tokens.__len__.return_value = 7
    attentions = attention_grads = tf.zeros([12, 12, 7, 7])
    recognizer.input_validation(example, attentions, attention_grads)

    with pytest.raises(ValueError) as info:
        attentions = attention_grads = tf.zeros([12, 12, 10, 10])
        recognizer.input_validation(example, attentions, attention_grads)


def test_get_indices():
    recognizer = AttentionGradientProduct  # Static method
    aspect_span = mock.MagicMock()
    aspect_span.tokens = ['CLS', 'this', 'is', 'a', 'test', 'SEP', 'test',
                          'SEP']
    cls_id, text_ids, aspect_id = recognizer.get_indices(aspect_span)
    assert cls_id == 0
    assert text_ids == [1, 2, 3, 4]
    assert aspect_id == 6


def test_scale():
    recognizer = AttentionGradientProduct  # Static method
    product = np.array([[1, -1, 5, 6],
                         [2, -1, 2, 1],
                         [0, -1, -5, 1]])
    normalized = np.round(recognizer.scale(product), decimals=2).tolist()
    assert normalized == [[0.17, -0.17, 0.83, 1.0],
                          [0.33, -0.17, 0.33, 0.17],
                          [0.0, -0.17, -0.83, 0.17]]


def test_get_key_mixtures():
    recognizer = AttentionGradientProduct  # Static method
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


def test_construct_patterns():
    recognizer = AttentionGradientProduct  # Static method
    aspect_span = mock.MagicMock()
    impacts = [1, 2, 3]
    mixtures = np.arange(12).reshape(3, 4).tolist()
    patterns = recognizer.construct_patterns(aspect_span, impacts, mixtures)
    assert len(patterns) == 3
    pattern_1, pattern_2, pattern_3 = patterns
    assert pattern_2.impact == 2
    assert pattern_2.weights == [4, 5, 6, 7]
