from unittest import mock

import pytest
import transformers
import numpy as np

from aspect_based_sentiment_analysis import LabeledExample
from aspect_based_sentiment_analysis import Sentiment
from aspect_based_sentiment_analysis.training import ClassifierDataset
np.random.seed(2)


@pytest.fixture
def tokenizer() -> transformers.PreTrainedTokenizer:
    return transformers.BertTokenizer.from_pretrained('bert-base-uncased')


def test_preprocess_batch(tokenizer):
    example_1 = LabeledExample(
        text='The breakfast was delicious, really great.',
        aspect='breakfast',
        sentiment=Sentiment.positive
    )
    example_2 = LabeledExample(
        text='The hotel is expensive.',
        aspect='hotel',
        sentiment=Sentiment.negative
    )
    dataset = ClassifierDataset(mock.Mock(), mock.Mock(), tokenizer)
    batch = dataset.preprocess_batch([example_1, example_2])
    target_labels = batch.target_labels.numpy()
    assert target_labels.shape == (2, 3)
    assert target_labels[0, -1] == 1
    # The first sentence contains 3 tokens more. So, second sentence should
    # be padded.
    attention_mask = batch.attention_mask.numpy()
    pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    assert np.allclose(attention_mask[1, -3:], pad_token_id)
