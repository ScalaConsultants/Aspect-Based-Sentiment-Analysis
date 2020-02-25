from unittest import mock
import pytest
import transformers
import numpy as np

from aspect_based_sentiment_analysis.data_types import Aspect
from aspect_based_sentiment_analysis.data_types import ClassifierExample
from aspect_based_sentiment_analysis.preprocessing \
    .classifier_model_input import ClassifierDataset
np.random.seed(2)


@pytest.fixture
def tokenizer() -> transformers.PreTrainedTokenizer:
    return transformers.BertTokenizer.from_pretrained('bert-base-uncased')


def test_preprocess_batch(tokenizer):
    example_1 = ClassifierExample(
        text='The breakfast was delicious, really great.',
        aspect=Aspect(name='breakfast', label=1)
    )
    example_2 = ClassifierExample(
        text='The hotel is expensive.',
        aspect=Aspect(name='hotel', label=-1)
    )
    dataset = ClassifierDataset(mock.Mock(), mock.Mock(), tokenizer)
    batch = dataset.preprocess_batch([example_1, example_2])
    assert np.allclose(batch.target_labels.numpy(), [1, -1])
    # The first sentence contains 3 tokens more. So, second sentence should
    # be padded.
    attention_mask = batch.attention_mask.numpy()
    pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    assert np.allclose(attention_mask[1, -3:], pad_token_id)
