import pytest
import numpy as np
import transformers
import tensorflow as tf

from aspect_based_sentiment_analysis import (
    BertABSClassifier,
    BertABSCConfig,
    CompletedTask,
    Pipeline,
    PredictedExample,
    Sentiment,
    Example,
    Task,
    Professor,
    load_examples,
    Review
)

np.random.seed(1)
tf.random.set_seed(1)


@pytest.fixture
def nlp() -> Pipeline:
    name = 'absa/classifier-rest-0.2'
    tokenizer = transformers.BertTokenizer.from_pretrained(name)
    # We pass a config explicitly (however, it can be downloaded automatically)
    config = BertABSCConfig.from_pretrained(name)
    model = BertABSClassifier.from_pretrained(name, config=config)
    professor = Professor()  # Without both pattern and reference recognizers.
    nlp = Pipeline(model, tokenizer, professor, text_splitter=None)
    return nlp


def test_call(nlp: Pipeline):
    text = ("We are great fans of Slack, but we wish the subscriptions "
            "were more accessible to small startups.")
    completed_task = nlp(text, aspects=['slack', 'price'])
    assert isinstance(completed_task, CompletedTask)
    slack, price = completed_task
    assert slack.sentiment == Sentiment.positive
    assert price.sentiment == Sentiment.negative


def test_preprocess(nlp: Pipeline):
    # We split a document into spans (in this case separated by the new line).
    nlp.text_splitter = lambda text: text.split('\n')
    raw_document = ("This is the test sentence 1.\n"
                    "This is the test sentence 2.\n"
                    "This is the test sentence 3.")
    task = nlp.preprocess(
        text=raw_document,
        aspects=['aspect_1', 'aspect_2'])
    assert isinstance(task, Task)
    assert len(task.subtasks) == 2
    assert list(task.subtasks) == ['aspect_1', 'aspect_2']
    assert len(task.examples) == 6
    assert task.indices == [(0, 3), (3, 6)]
    subtask_1, subtask_2 = task
    assert subtask_1.text == subtask_2.text == raw_document
    assert subtask_1.aspect == 'aspect_1'
    assert len(subtask_1.examples) == 3


def test_encode(nlp: Pipeline):
    text_1 = ("We are great fans of Slack, but we wish the subscriptions "
              "were more accessible to small startups.")
    text_2 = "We are great fans of Slack"
    aspect = "Slack"

    examples = [Example(text_1, aspect), Example(text_2, aspect)]
    tokenized_examples = nlp.tokenize(examples)
    input_batch = nlp.encode(tokenized_examples)
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


def test_predict(nlp: Pipeline):
    text_1 = ("We are great fans of Slack, but we wish the subscriptions "
              "were more accessible to small startups.")
    text_2 = "We are great fans of Slack"
    aspect = "Slack"
    examples = [Example(text_1, aspect), Example(text_2, aspect)]
    tokenized_examples = nlp.tokenize(examples)
    input_batch = nlp.encode(tokenized_examples)
    output_batch = nlp.predict(input_batch)
    assert output_batch.scores.shape == [2, 3]
    assert output_batch.hidden_states.shape == [2, 13, 25, 768]
    assert output_batch.attentions.shape == [2, 12, 12, 25, 25]
    assert output_batch.attention_grads.shape == [2, 12, 12, 25, 25]
    scores = output_batch.scores.numpy()
    assert np.argmax(scores, axis=-1).tolist() == [2, 2]


def test_review(nlp: Pipeline):
    text_1 = ("We are great fans of Slack, but we wish the subscriptions "
              "were more accessible to small startups.")
    text_2 = "The Slack often has bugs."
    aspect = "slack"
    examples = [Example(text_1, aspect), Example(text_2, aspect)]

    tokenized_examples = nlp.tokenize(examples)
    input_batch = nlp.encode(tokenized_examples)
    output_batch = nlp.predict(input_batch)
    predictions = nlp.review(tokenized_examples, output_batch)
    predictions = list(predictions)

    labeled_1, labeled_2 = predictions
    assert labeled_1.sentiment == Sentiment.positive
    assert labeled_2.sentiment == Sentiment.negative
    assert isinstance(labeled_1, PredictedExample)
    assert isinstance(labeled_1.scores, list)
    assert isinstance(labeled_1.review, Review)
    assert not labeled_1.review.is_reference
    assert not labeled_1.review.patterns


def test_postprocess(nlp: Pipeline):
    text = ("We are great fans of Slack.\n"
            "The Slack often has bugs.\n"
            "best of all is the warm vibe")
    # Define a naive text_splitter.
    nlp.text_splitter = lambda text: text.split('\n')

    task = nlp.preprocess(text, aspects=['slack', 'price'])
    predictions = nlp.transform(task.examples)
    completed_task = nlp.postprocess(task, predictions)

    assert len(completed_task.examples) == 6
    assert completed_task.indices == [(0, 3), (3, 6)]

    slack, price = completed_task
    assert slack.text == price.text == text
    # The sentiment among fragments are different. We normalize scores.
    assert np.allclose(slack.scores, [0.03, 0.48, 0.48], atol=0.01)
    # Please note that there is a problem with the neutral sentiment.
    assert np.allclose(price.scores, [0.02, 0.49, 0.49], atol=0.01)


def test_evaluate(nlp: Pipeline):
    examples = load_examples(
        dataset='semeval',
        domain='restaurant',
        test=True)
    metric = tf.metrics.Accuracy()
    result = nlp.evaluate(examples[:40], metric, batch_size=10)
    result = result.numpy()
    # The model predicts the first 40 labels perfectly.
    assert result == 1
    result = nlp.evaluate(examples[40:50], metric, batch_size=10)
    assert np.isclose(result, 0.98)
