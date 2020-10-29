from unittest import mock

import numpy as np
import tensorflow as tf

import aspect_based_sentiment_analysis as absa
from aspect_based_sentiment_analysis import BasicPatternRecognizer
from aspect_based_sentiment_analysis import BasicReferenceRecognizer
from aspect_based_sentiment_analysis import Example
from aspect_based_sentiment_analysis import PredictedExample
from aspect_based_sentiment_analysis import alignment
from aspect_based_sentiment_analysis import Output
from aspect_based_sentiment_analysis import Pattern


def test_basic_reference_recognizer():
    text = 'the automobile is so cool and the service is prompt and curious.'
    examples = [Example(text, 'breakfast'), Example(text, 'service'), Example(text, 'car')]
    recognizer = BasicReferenceRecognizer(weights=(-0.025, 44))
    nlp = absa.load('absa/classifier-rest-0.2', reference_recognizer=recognizer)
    predictions = nlp.transform(examples)
    prediction_1, prediction_2, prediction_3 = predictions
    assert not prediction_1.review.is_reference
    assert prediction_2.review.is_reference
    assert prediction_3.review.is_reference


def test_basic_reference_recognizer_from_pretrained():
    name = 'absa/basic_reference_recognizer-rest-0.1'
    recognizer = BasicReferenceRecognizer.from_pretrained(name)
    assert np.allclose(recognizer.weights, [-0.024, 44.443], atol=0.001)
    name = 'absa/basic_reference_recognizer-lapt-0.1'
    recognizer = BasicReferenceRecognizer.from_pretrained(name)
    assert np.allclose(recognizer.weights, [-0.175, 40.165], atol=0.001)


def test_basic_reference_recognizer_transform():
    h = tf.constant([[0, 1, 0],
                     [0, 1, 0],
                     [0, 0, 0],
                     [4, 3, 0]], dtype=float)
    hidden_states = tf.reshape(h, [1, 4, 3])
    text_mask = [True, True, False, False]
    aspect_mask = [False, False, False, True]
    similarity = BasicReferenceRecognizer.transform(
        hidden_states, text_mask, aspect_mask)
    expected = np.array([0, 1, 0]) @ np.array([.8, .6, 0])
    assert expected == 0.6
    assert np.isclose(similarity, expected, atol=1e-7, rtol=0)


def test_basic_reference_recognizer_aspect_subtoken_masks():
    e = mock.Mock()
    e.text_subtokens = list('abc')
    e.aspect_subtokens = list('de')
    e.subtokens = ['cls', *e.text_subtokens, 'sep', *e.aspect_subtokens, 'sep']
    text_mask, aspect_mask = BasicReferenceRecognizer.text_aspect_subtoken_masks(e)
    assert text_mask == [False, True, True, True, False, False, False, False]
    assert aspect_mask == [False, False, False, False, False, True, True, False]


def test_basic_pattern_recognizer():
    text = ("We are great fans of Slack, but we wish the subscriptions "
            "were more accessible to small startups.")
    example = Example(text, aspect='price')
    recognizer = BasicPatternRecognizer()
    nlp = absa.load('absa/classifier-rest-0.2', pattern_recognizer=recognizer)
    predictions = nlp.transform([example])
    prediction = next(predictions)
    assert isinstance(prediction, PredictedExample)
    assert not prediction.review.is_reference
    assert len(prediction.review.patterns) == 5
    pattern, *_ = prediction.review.patterns
    assert pattern.importance == 1
    assert list(zip(pattern.tokens, pattern.weights)) == \
        [('we', 0.25), ('are', 0.44), ('great', 0.88), ('fans', 0.92),
         ('of', 0.2), ('slack', 0.27), (',', 0.46), ('but', 1.0), ('we', 0.36),
         ('wish', 0.95), ('the', 0.16), ('subscriptions', 0.39), ('were', 0.23),
         ('more', 0.33), ('accessible', 0.24), ('to', 0.13), ('small', 0.14),
         ('startups', 0.24), ('.', 0.28)]


def test_basic_pattern_recognizer_text_token_indices():
    example = mock.Mock()
    example.text_tokens = ['we', 'are', 'soooo', 'great', 'fans', 'of', 'slack']
    example.tokens = ['[CLS]', *example.text_tokens, '[SEP]', 'slack', '[SEP]']
    mask = BasicPatternRecognizer.text_tokens_mask(example)
    assert mask == [False, True, True, True, True, True, True, True, False, False, False]


def test_basic_pattern_recognizer_transform(monkeypatch):
    recognizer = BasicPatternRecognizer(
        max_patterns=3, is_scaled=False, is_rounded=False)
    monkeypatch.setattr(alignment, "merge_tensor", lambda x, **kwargs: x)
    x = tf.constant([[0, 1, 2, 0],
                     [0, 1, 3, 0],
                     [0, 2, 4, 0],
                     [0, 0, 0, 0]], dtype=float)
    x = tf.reshape(x, [1, 1, 4, 4])
    text_mask = [False, True, True, False]
    output = Output(
        scores=None,
        attentions=x,
        attention_grads=tf.ones_like(x, dtype=float),
        hidden_states=None)
    w, pattern_vectors = recognizer.transform(
        output, text_mask, token_subtoken_alignment=None)
    assert w.tolist() == [0.5, 1]
    assert pattern_vectors.tolist() == [[1., 1.],  # [3 3] / [3]
                                        [.5, 1.]]  # [2 4] / [4]

    recognizer = BasicPatternRecognizer(
        max_patterns=3, is_scaled=True, is_rounded=True)
    w, pattern_vectors = recognizer.transform(
        output, text_mask, token_subtoken_alignment=None)
    assert w.tolist() == [0.5, 1]
    assert pattern_vectors.tolist() == [[.5, .5],
                                        [.5, 1.]]


def test_basic_pattern_recognizer_build_patterns():
    recognizer = BasicPatternRecognizer(
        max_patterns=3, is_scaled=False, is_rounded=False)
    w = np.array([0, 3, 2, 0, 1])
    pattern_vectors = np.array([[0, 0, 0],
                                [1, 1, 1],
                                [2, 2, 2],
                                [3, 3, 3],
                                [4, 4, 4]])
    exemplary_tokens = list('abcde')
    patterns = recognizer.build_patterns(w, exemplary_tokens, pattern_vectors)
    assert len(patterns) == 3
    assert all(p.tokens == exemplary_tokens for p in patterns)
    assert [p.importance for p in patterns] == [3, 2, 1]
    assert [p.weights for p in patterns] == [[1, 1, 1],
                                             [2, 2, 2],
                                             [4, 4, 4]]
    w = np.array([1, 3, 2, 0, 1])
    patterns = recognizer.build_patterns(w, exemplary_tokens, pattern_vectors)
    assert len(patterns) == 3
    *_, pattern_3 = patterns
    assert pattern_3.importance == 1
    assert pattern_3.weights == [0, 0, 0]


def test_predict_key_set():
    tokens = list('abcd')
    weights = [[0, 0, 0, 3],
               [1, 0, 1, 0],
               [1, 0, 0, 0],
               [2, 0, 1, 0]]
    patterns = [Pattern(None, tokens, w) for w in weights]
    key_set_candidate = absa.predict_key_set(patterns, n=2)
    assert key_set_candidate == {0, 3}
