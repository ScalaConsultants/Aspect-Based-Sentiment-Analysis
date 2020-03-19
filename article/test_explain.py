import pytest
import numpy as np
import tensorflow as tf

from article.explain import (
    Template,
    merge_input_attentions,
    calculate_activation_means,
    get_rule
)

np.random.seed(1)


@pytest.fixture
def template() -> Template:
    template = Template(
        text="don't go alone---even two people isn't enough for the whole "
             "experience, with pickles and a selection of meats and seafoods.",
        aspect='pickles',
        aspect_tokens=['pickles'],
        tokens=['[CLS]', 'don', "'", 't', 'go', 'alone', '-', '-', '-', 'even',
                'two', 'people', 'isn', "'", 't', 'enough', 'for', 'the',
                'whole', 'experience', ',', 'with', 'pickles', 'and', 'a',
                'selection', 'of', 'meats', 'and', 'seafoods', '.', '[SEP]',
                'pickles', '[SEP]'],
        sub_tokens=['[CLS]', 'don', "'", 't', 'go', 'alone', '-', '-', '-',
                    'even', 'two', 'people', 'isn', "'", 't', 'enough', 'for',
                    'the', 'whole', 'experience', ',', 'with', 'pick', '##les',
                    'and', 'a', 'selection', 'of', 'meat', '##s', 'and',
                    'seafood', '##s', '.', '[SEP]', 'pick', '##les', '[SEP]'],
        alignment=[[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11],
                   [12], [13], [14], [15], [16], [17], [18], [19], [20], [21],
                   [22, 23], [24], [25], [26], [27], [28, 29], [30], [31, 32],
                   [33], [34], [35, 36], [37]]
    )
    return template


def test_get_rule():
    tokens = ['[CLS]', 'ab', 'bc', '[SEP]', 'cd', 'de', '[SEP]']
    aspect_tokens = ['cd', 'de']
    rule = get_rule('ALL', aspect_tokens)
    mask = [rule(t) for t in tokens]
    assert all(mask)
    rule = get_rule('NON-SPECIAL', aspect_tokens)
    mask = [rule(t) for t in tokens]
    assert mask == [False, True, True, False, False, False, False]
    rule = get_rule('CLS', aspect_tokens)
    mask = [rule(t) for t in tokens]
    assert mask[0] and not any(mask[1:])
    rule = get_rule('SEP', aspect_tokens)
    mask = [rule(t) for t in tokens]
    assert mask == [False, False, False, True, False, False, True]
    rule = get_rule('ASPECT', aspect_tokens)
    mask = [rule(t) for t in tokens]
    assert mask == [False, False, False, False, True, True, False]


def test_calculate_activation_means(template: Template):
    n = len(template.alignment)
    logits = np.random.randint(0, 10, size=[12, 12, n, n]).astype(float)
    α = tf.nn.softmax(logits, axis=-1).numpy()

    pattern = ('SEP', 'CLS')
    averages = calculate_activation_means(α, template, pattern)

    layer, head = 5, 7  # Randomly selected layer and head.
    assert averages[layer, head] == np.mean(α[layer, head, [31, 33], 0])


def test_get_word_attentions(template: Template):
    # Set up fake attentions
    n = len(template.sub_tokens)
    attentions = np.zeros([12, 12, 53, 53])
    logits = np.random.randint(0, 10, size=[12, 12, n, n]).astype(float)
    attentions[:, :, :n, :n] = tf.nn.softmax(logits, axis=-1).numpy()
    assert np.isclose(attentions[0, 0, 0, :].sum(), 1.0)
    assert np.isclose(attentions[0, 0, n, :].sum(), 0.0)

    n_align = len(template.alignment)
    α = merge_input_attentions(attentions, template.alignment)
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
