import pytest

from aspect_based_sentiment_analysis import (
    load,
    load_examples,
    LabeledExample,
    Sentiment
)
from aspect_based_sentiment_analysis.loads import NotFound


def test_load():
    nlp = load('absa/classifier-rest-0.2')
    text = ("We are great fans of Slack, but we wish the subscriptions "
            "were more accessible to small startups.")
    slack, price = nlp(text, aspects=['slack', 'price'])
    assert slack.sentiment == Sentiment.positive
    assert price.sentiment == Sentiment.negative


def test_load_semeval_classifier_examples():
    examples = load_examples(
        dataset='semeval',
        domain='laptop',
        test=False
    )
    assert len(examples) == 2313
    assert isinstance(examples[0], LabeledExample)

    test_examples = load_examples(
        dataset='semeval',
        domain='laptop',
        test=True
    )
    assert len(test_examples) == 638
    assert isinstance(test_examples[0], LabeledExample)

    texts = {e.text for e in examples}
    test_texts = {e.text for e in test_examples}
    # Interesting, there is a one sentence which appears in both
    # train and test set.
    assert len(texts & test_texts) == 1

    examples = load_examples(
        dataset='semeval',
        domain='restaurant',
        test=False
    )
    assert len(examples) == 3602
    assert isinstance(examples[0], LabeledExample)

    test_examples = load_examples(
        dataset='semeval',
        domain='restaurant',
        test=True
    )
    assert len(test_examples) == 1120
    assert isinstance(test_examples[0], LabeledExample)

    texts = {e.text for e in examples}
    test_texts = {e.text for e in test_examples}
    # Strange, there is also a one sentence in a restaurant dataset
    # which appears in both train and test set.
    assert len(texts & test_texts) == 1

    with pytest.raises(NotFound):
        load_examples(dataset='mistake')
