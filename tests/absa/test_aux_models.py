import pytest
import tensorflow as tf
from aspect_based_sentiment_analysis import TokenizedExample
from aspect_based_sentiment_analysis import BasicPatternRecognizer
from aspect_based_sentiment_analysis import alignment
from aspect_based_sentiment_analysis import Output


@pytest.fixture
def example() -> TokenizedExample:
    example = TokenizedExample(
        text='We are soooo great fans of Slack',
        text_tokens=['we', 'are', 'soooo', 'great', 'fans', 'of', 'slack'],
        text_subtokens=['we', 'are', 'soo', '##oo', 'great',
                        'fans', 'of', 'slack'],
        aspect='Slaeck',
        aspect_tokens=['slaeck'],
        aspect_subtokens=['sl', '##ae', '##ck'],
        tokens=['[CLS]', 'we', 'are', 'soooo', 'great', 'fans', 'of', 'slack',
                '[SEP]', 'slaeck', '[SEP]'],
        subtokens=['[CLS]', 'we', 'are', 'soo', '##oo', 'great', 'fans',
                   'of', 'slack', '[SEP]', 'sl', '##ae', '##ck', '[SEP]'],
        alignment=[[0], [1], [2], [3, 4], [5], [6], [7],
                   [8], [9], [10, 11, 12], [13]])
    return example


def test_basic_pattern_recognizer_text_token_indices(example):
    mask = BasicPatternRecognizer.text_tokens_mask(example)
    assert mask == [False, True, True, True, True, True, True, True, False,
                    False, False]


def test_basic_pattern_recognizer_transform(monkeypatch):
    recognizer = BasicPatternRecognizer(
        max_patterns=3, is_pattern_scaled=False, is_pattern_rounded=False)
    monkeypatch.setattr(alignment, "merge_tensor", lambda x, **kargs: x)
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
        max_patterns=3, is_pattern_scaled=True, is_pattern_rounded=True)
    w, pattern_vectors = recognizer.transform(
        output, text_mask, token_subtoken_alignment=None)
    assert w.tolist() == [0.5, 1]
    assert pattern_vectors.tolist() == [[.5, .5],
                                        [.5, 1.]]
