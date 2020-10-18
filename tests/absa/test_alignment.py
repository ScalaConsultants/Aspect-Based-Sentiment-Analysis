import pytest
import numpy as np
import tensorflow as tf
import transformers
from transformers import BertTokenizer

from aspect_based_sentiment_analysis import TokenizedExample
from aspect_based_sentiment_analysis import make_alignment
from aspect_based_sentiment_analysis import merge_tensor
from aspect_based_sentiment_analysis import tokenize

np.random.seed(1)


@pytest.fixture
def example() -> TokenizedExample:
    example = TokenizedExample(
        text='We are soooo great fans of Slack',
        text_tokens=['we', 'are', 'soooo', 'great', 'fans', 'of', 'slack'],
        text_subtokens=['we', 'are', 'soo', '##oo', 'great',
                        'fans', 'of', 'slack'],
        aspect='Slaeck',  # Here is an intentional typo.
        aspect_tokens=['slaeck'],
        aspect_subtokens=['sl', '##ae', '##ck'],
        tokens=['[CLS]', 'we', 'are', 'soooo', 'great', 'fans', 'of', 'slack',
                '[SEP]', 'slaeck', '[SEP]'],
        subtokens=['[CLS]', 'we', 'are', 'soo', '##oo', 'great', 'fans',
                   'of', 'slack', '[SEP]', 'sl', '##ae', '##ck', '[SEP]'],
        alignment=[[0], [1], [2], [3, 4], [5], [6], [7],
                   [8], [9], [10, 11, 12], [13]])
    return example


@pytest.fixture
def tokenizer() -> BertTokenizer:
    name = 'bert-base-uncased'
    tokenizer = transformers.BertTokenizer.from_pretrained(name)
    return tokenizer


def test_tokenize(example: TokenizedExample, tokenizer: BertTokenizer):
    test_example = tokenize(tokenizer, example.text, example.aspect)
    assert test_example == example


def test_make_alignment(example: TokenizedExample, tokenizer: BertTokenizer):
    wordpiece_tokenizer = tokenizer.wordpiece_tokenizer
    sub_tokens, alignment = make_alignment(wordpiece_tokenizer, example.tokens)
    assert sub_tokens == example.subtokens
    assert alignment == example.alignment


def test_merge(example: TokenizedExample):
    # Set up fake attentions
    n = len(example.subtokens)
    attentions = np.zeros([12, 12, 53, 53])
    logits = np.random.randint(0, 10, size=[12, 12, n, n]).astype(float)
    attentions[:, :, :n, :n] = tf.nn.softmax(logits, axis=-1)
    attentions = tf.convert_to_tensor(attentions)
    assert np.isclose(tf.reduce_sum(attentions[0, 0, 0, :]), 1.0)
    assert np.isclose(tf.reduce_sum(attentions[0, 0, n, :]), 0.0)

    n_align = len(example.alignment)
    α = merge_tensor(attentions, example.alignment)

    # Convert to numpy arrays.
    attentions = attentions.numpy()
    α = α.numpy()

    assert α.shape == (12, 12, n_align, n_align)
    # Still, attention distributions should sum to one.
    assert np.allclose(α.sum(axis=-1), 1)

    layer, head = 5, 7  # Randomly selected layer and head.
    # We choose an arbitrary (not divided) token. At the beginning, the indices
    # are the same. Attentions between not divided tokens should be unchanged.
    assert attentions[layer, head, 1, 2] == α[layer, head, 1, 2]

    # For attention _to_ a split-up word, we sum up the attention weights
    # over its tokens. Note, that the `k` means the index in the alignments.
    # Here, we test an attention between "slaeck" and "we" (index 1).
    k, ij = 9, [10, 11, 12]
    attention_to = np.sum(attentions[layer, head, 1, ij])
    assert attention_to == α[layer, head, 1, k]

    # For attention _from_ a split-up word, we take the mean of the attention
    # weights over its tokens. Here, we choose a different split-up word
    # "soooo" and "fans" (an index 6 before and 5 after transformation)
    k, ij = 3, [3, 4]
    attention_from = np.mean(attentions[layer, head, ij, 6])
    assert attention_from == α[layer, head, k, 5]
