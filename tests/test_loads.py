import pytest
from aspect_based_sentiment_analysis import loads
from aspect_based_sentiment_analysis.training import ClassifierExample


def test_load_semeval_classifier_examples():
    examples = loads.load_classifier_examples(
        dataset='semeval',
        domain='laptop',
        test=False
    )
    assert len(examples) == 2313
    assert isinstance(examples[0], ClassifierExample)

    test_examples = loads.load_classifier_examples(
        dataset='semeval',
        domain='laptop',
        test=True
    )
    assert len(test_examples) == 638
    assert isinstance(test_examples[0], ClassifierExample)

    texts = {e.text for e in examples}
    test_texts = {e.text for e in test_examples}
    # Interesting, there is a one sentence which appears in both
    # train and test set.
    assert len(texts & test_texts) == 1

    examples = loads.load_classifier_examples(
        dataset='semeval',
        domain='restaurant',
        test=False
    )
    assert len(examples) == 3602
    assert isinstance(examples[0], ClassifierExample)

    test_examples = loads.load_classifier_examples(
        dataset='semeval',
        domain='restaurant',
        test=True
    )
    assert len(test_examples) == 1120
    assert isinstance(test_examples[0], ClassifierExample)

    texts = {e.text for e in examples}
    test_texts = {e.text for e in test_examples}
    # Strange, there is also a one sentence in a restaurant dataset
    # which appears in both train and test set.
    assert len(texts & test_texts) == 1

    with pytest.raises(loads.NotFound):
        loads.load_classifier_examples(dataset='mistake')
