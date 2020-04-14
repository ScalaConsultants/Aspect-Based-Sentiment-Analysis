import pytest
import numpy as np
import tensorflow as tf
import transformers
from transformers import BertTokenizer

from aspect_based_sentiment_analysis import TokenizedExample
from aspect_based_sentiment_analysis import make_alignment
from aspect_based_sentiment_analysis import merge_input_attentions
np.random.seed(1)


@pytest.fixture
def example() -> TokenizedExample:
    example = TokenizedExample(
        text="don't go alone---even two people isn't enough for the whole "
             "experience, with pickles and a selection of meats and seafoods.",
        text_tokens=['don', "'", 't', 'go', 'alone', '-', '-', '-', 'even',
                     'two', 'people', 'isn', "'", 't', 'enough', 'for', 'the',
                     'whole', 'experience', ',', 'with', 'pickles', 'and', 'a',
                     'selection', 'of', 'meats', 'and', 'seafoods', '.'],
        tokens=['[CLS]', 'don', "'", 't', 'go', 'alone', '-', '-', '-', 'even',
                'two', 'people', 'isn', "'", 't', 'enough', 'for', 'the',
                'whole', 'experience', ',', 'with', 'pickles', 'and', 'a',
                'selection', 'of', 'meats', 'and', 'seafoods', '.', '[SEP]',
                'pickles', '[SEP]'],
        subtokens=['[CLS]', 'don', "'", 't', 'go', 'alone', '-', '-', '-',
                    'even', 'two', 'people', 'isn', "'", 't', 'enough', 'for',
                    'the', 'whole', 'experience', ',', 'with', 'pick', '##les',
                    'and', 'a', 'selection', 'of', 'meat', '##s', 'and',
                    'seafood', '##s', '.', '[SEP]', 'pick', '##les', '[SEP]'],
        alignment=[[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11],
                   [12], [13], [14], [15], [16], [17], [18], [19], [20], [21],
                   [22, 23], [24], [25], [26], [27], [28, 29], [30], [31, 32],
                   [33], [34], [35, 36], [37]],
        aspect='pickles',
        aspect_tokens=['pickles']
    )
    return example


@pytest.fixture
def tokenizer() -> BertTokenizer:
    name = 'bert-base-uncased'
    tokenizer = transformers.BertTokenizer.from_pretrained(name)
    return tokenizer


def test_make_alignment(example: TokenizedExample, tokenizer: BertTokenizer):
    wordpiece_tokenizer = tokenizer.wordpiece_tokenizer
    sub_tokens, alignment = make_alignment(wordpiece_tokenizer,
                                           example.tokens)
    assert sub_tokens == example.subtokens
    assert alignment == example.alignment


def test_merge_input_attentions(example: TokenizedExample):
    # Set up fake attentions
    n = len(example.subtokens)
    attentions = np.zeros([12, 12, 53, 53])
    logits = np.random.randint(0, 10, size=[12, 12, n, n]).astype(float)
    attentions[:, :, :n, :n] = tf.nn.softmax(logits, axis=-1).numpy()
    assert np.isclose(attentions[0, 0, 0, :].sum(), 1.0)
    assert np.isclose(attentions[0, 0, n, :].sum(), 0.0)

    n_align = len(example.alignment)
    α = merge_input_attentions(attentions, example.alignment)
    assert α.shape == (12, 12, n_align, n_align)
    # Still, attention distributions should sum to one.
    assert np.allclose(α.sum(axis=-1), 1)

    layer, head = 5, 7  # Randomly selected layer and head.
    # We choose an arbitrary (not divided) token. At the beginning, the indices
    # are the same. Attentions between not divided tokens should be unchanged.
    assert attentions[layer, head, 4, 10] == α[layer, head, 4, 10]

    # For attention _to_ a split-up word, we sum up the attention weights
    # over its tokens. Note, that the `k` means the index in the alignments.
    k, ij = 32, [35, 36]
    attention_to = np.sum(attentions[layer, head, 4, ij])
    assert attention_to == α[layer, head, 4, k]

    # For attention _from_ a split-up word, we take the mean of the attention
    # weights over its tokens.
    ij, k = [28, 29], 27  # We can choose different split-up word.
    attention_from = np.mean(attentions[layer, head, ij, 6])
    assert attention_from == α[layer, head, k, 6]
