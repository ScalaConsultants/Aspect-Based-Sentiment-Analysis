from unittest import mock

import pytest
import transformers
import numpy as np

from aspect_based_sentiment_analysis.training import (
    LanguageModelExample,
    LanguageModelDataset
)


@pytest.fixture
def tokenizer() -> transformers.BertTokenizer:
    return transformers.BertTokenizer.from_pretrained('bert-base-uncased')


def test_examples_generator(tokenizer):
    # Let's define a single task.
    doc = [f'This is an arbitrary sentence {i}.' for i in range(10)]
    document_store = mock.MagicMock()
    document_store.__iter__.return_value = iter([doc])
    random_doc = [f'This is a random sentence {i}.' for i in range(10)]
    document_store.sample_doc.return_value = random_doc

    # Turn off short_seq_prob. The batch_size does not play
    # a role in generating example.
    dataset = LanguageModelDataset(document_store,
                                   batch_size=0,
                                   tokenizer=tokenizer,
                                   max_num_tokens=10,
                                   short_seq_prob=0)

    generator = dataset.examples_generator()
    examples = list(generator)

    assert len(tokenizer.tokenize(doc[0])) == 7
    # Each sentence in a task has 7 tokens, and our the max number of
    # tokens equals 10, so two segments is enough to build a valid tokens
    # pair. We can construct 5 pairs (in the case when the flag `is_next`
    # is true).
    assert len(examples) >= 5


def test_preprocess_batch(tokenizer):
    np.random.seed(2)  # Make sure
    example_1 = LanguageModelExample(
        text_a='This is an arbitrary sentence A',
        text_b='This is an arbitrary sentence B',
        is_next=True
    )
    example_2 = LanguageModelExample(
        text_a='This is the long long long long sentence A',
        text_b='This is the long long long long sentence B',
        is_next=False
    )
    dataset = LanguageModelDataset(mock.Mock(),
                                   mock.Mock(),
                                   tokenizer=tokenizer,
                                   max_num_tokens=mock.Mock(),
                                   mlm_probability=0.2,
                                   short_seq_prob=0)
    batch = dataset.preprocess_batch([example_1, example_2])
    assert batch.token_ids.shape == (2, 21)
    # We do padding to the longest sequence.
    attention_mask = batch.attention_mask.numpy()
    pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    assert np.allclose(attention_mask[0, -6:], pad_token_id)
    assert np.allclose(batch.target_is_next.numpy(), [True, False])

    inputs = batch.token_ids.numpy()
    batch_size, seq_length = inputs.shape
    assert np.allclose(inputs[:, 0], tokenizer.cls_token_id)
    # We check the separator tokens in the second example
    # (the first example has also pad tokens at the end)
    assert np.allclose(inputs[1, [seq_length//2, -1]], tokenizer.sep_token_id)

    # We make sure, that targets are not in the input, outside of randomization
    targets = batch.target_masked_token_ids.numpy()
    mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    assert set(inputs[targets != -100]) == {mask_token_id}
    # More sophisticated checks are implemented in the `mask_tokens`
    # function in the `language_model_functions` module.


def test_target_length():
    dataset = LanguageModelDataset(mock.Mock(),
                                   mock.Mock(),
                                   mock.Mock(),
                                   max_num_tokens=512,
                                   short_seq_prob=0.2)
    N = int(10e4)
    target_lengths = np.array([dataset.target_length() for i in range(N)])
    assert round(np.sum(target_lengths < 512) / N, 2) == 0.2
    assert round(np.sum(target_lengths == 512) / N, 2) == 0.8


def test_generate_random_segments():
    document_store = mock.MagicMock()
    document_store.__len__.return_value = 10
    random_doc = [f'This is a random sentence {i}.' for i in range(10)]
    document_store.sample_doc.return_value = random_doc

    tokenizer = mock.MagicMock()
    tokenizer.tokenize = lambda sentence: sentence

    dataset = LanguageModelDataset(document_store,
                                   mock.Mock(),
                                   tokenizer,
                                   mock.Mock(),
                                   mock.Mock(),
                                   mock.Mock())

    # We check if our generator returns segments
    # which are trimmed correctly.
    generator = dataset.generate_random_segments(
        doc_index=0, start_indices=iter([0, 0.5, 1])
    )
    random_segments = list(next(generator))
    assert len(random_segments) == 10
    random_segments = list(next(generator))
    assert len(random_segments) == 5
    assert random_segments[0] == 'This is a random sentence 5.'
    random_segments = list(next(generator))
    assert len(random_segments) == 0
