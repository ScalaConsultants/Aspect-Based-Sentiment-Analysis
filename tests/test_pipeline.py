import pytest
import mock
import numpy as np
import transformers
import tensorflow as tf

from aspect_based_sentiment_analysis import (
    BertABSClassifier,
    BertPipeline,
    Sentiment
)
from aspect_based_sentiment_analysis.probing import (
    AttentionPatternRecognizer
)
np.random.seed(1)
tf.random.set_seed(1)


@pytest.fixture
def name():
    return 'absa/classifier-rest-0.1'


@pytest.fixture
def tokenizer(name):
    tokenizer = transformers.BertTokenizer.from_pretrained(name)
    return tokenizer


@pytest.mark.skip
def test_integration(name, tokenizer):
    model = BertABSClassifier.from_pretrained(
        name,
        output_attentions=True,
        output_hidden_states=True
    )
    nlp = BertPipeline(model, tokenizer)
    text = ("We are great fans of Slack, but we wish the subscriptions "
            "were more accessible to small startups.")
    slack, price = nlp(text, aspects=['slack', 'price'])


@pytest.mark.skip
def test_get_document(tokenizer):
    model = mock.MagicMock()
    naive_sentencizer = lambda text: text.split('\n')
    nlp = BertPipeline(model, tokenizer, sentencizer=naive_sentencizer)
    raw_document = ("This is the test sentence 1.\n"
                    "This is the test sentence 2.\n"
                    "This is the test sentence 3.")
    document = nlp.get_document(
        text=raw_document,
        aspects=['aspect_1', 'aspect_2']
    )
    assert len(document.aspect_docs) == 2
    assert list(document.aspect_docs) == ['aspect_1', 'aspect_2']
    assert len(document.batch) == 6
    assert document.indices == [(0, 3), (3, 6)]
    aspect_1, aspect_2 = document
    assert aspect_1.text == aspect_2.text == raw_document
    assert aspect_1.aspect == 'aspect_1'
    assert len(aspect_1.aspect_spans) == 3


@pytest.mark.skip
def test_batch(tokenizer):
    model = mock.MagicMock()
    nlp = BertPipeline(model, tokenizer)
    text_1 = ("We are great fans of Slack, but we wish the subscriptions "
              "were more accessible to small startups.")
    text_2 = "We are great fans of Slack"
    aspect = "Slack"
    aspect_spans = nlp.preprocess(pairs=[(text_1, aspect), (text_2, aspect)])
    input_batch = nlp.batch(aspect_spans)
    assert isinstance(input_batch.token_ids, tf.Tensor)
    # 101 the CLS token, 102 the SEP tokens.
    token_ids = input_batch.token_ids.numpy()
    values = [101, 2057, 2024, 2307, 4599, 1997, 19840, 102, 19840, 102]
    assert token_ids[1, :10].tolist() == values
    assert token_ids[0, :7].tolist() == values[:7]
    # The second sequence should be padded (shorter),
    # and attention mask should be set.
    assert np.allclose(token_ids[1, 10:], 0)
    attention_mask = input_batch.attention_mask.numpy()
    assert np.allclose(attention_mask[1, 10:], 0)
    # Check how the tokenizer marked the segments.
    token_type_ids = input_batch.token_type_ids.numpy()
    assert token_type_ids[0, -2:].tolist() == [1, 1]
    assert np.allclose(token_type_ids[0, :-2], 0)


@pytest.mark.skip
def test_predict(name, tokenizer):
    model = BertABSClassifier.from_pretrained(
        name,
        output_attentions=True,
        output_hidden_states=True
    )
    nlp = BertPipeline(model, tokenizer)
    text_1 = ("We are great fans of Slack, but we wish the subscriptions "
              "were more accessible to small startups.")
    text_2 = "We are great fans of Slack"
    aspect = "Slack"
    aspect_spans = nlp.preprocess(pairs=[(text_1, aspect), (text_2, aspect)])
    input_batch = nlp.batch(aspect_spans)
    output_batch = nlp.predict(input_batch)
    assert output_batch.scores.shape == [2, 3]
    assert output_batch.hidden_states.shape == [2, 13, 23, 768]
    assert output_batch.attentions.shape == [2, 12, 12, 23, 23]
    assert output_batch.attention_grads.shape == [2, 12, 12, 23, 23]
    scores = output_batch.scores.numpy()
    assert np.argmax(scores, axis=-1).tolist() == [2, 2]


@pytest.mark.skip
def test_label(name, tokenizer):
    model = BertABSClassifier.from_pretrained(
        name,
        output_attentions=True,
        output_hidden_states=True
    )
    pattern_recognizer = AttentionPatternRecognizer()
    nlp = BertPipeline(
        model, tokenizer,
        pattern_recognizer=pattern_recognizer
    )
    text_1 = ("We are great fans of Slack, but we wish the subscriptions "
              "were more accessible to small startups.")
    text_2 = "The Slack often has bugs."
    text_3 = "best of all is the warm vibe"
    aspect = "slack"
    pairs = [(text_1, aspect), (text_2, aspect), (text_3, aspect)]

    aspect_spans = nlp.preprocess(pairs)
    input_batch = nlp.batch(aspect_spans)
    output_batch = nlp.predict(input_batch)
    aspect_span_labeled = nlp.label(aspect_spans, output_batch)
    aspect_span_labeled = list(aspect_span_labeled)
    labeled_1, labeled_2, labeled_3 = aspect_span_labeled
    assert labeled_1.sentiment == Sentiment.positive
    assert labeled_2.sentiment == Sentiment.negative
    assert np.argmax(labeled_1.aspect_representation.look_at) == 5
    assert np.argmax(labeled_2.aspect_representation.look_at) == 1

    # We need to calibrate the model. The prediction should be neutral.
    # In fact, the model does not recognize the aspect.
    assert labeled_3.sentiment == Sentiment.positive
    assert np.allclose(labeled_3.aspect_representation.look_at, 0)
