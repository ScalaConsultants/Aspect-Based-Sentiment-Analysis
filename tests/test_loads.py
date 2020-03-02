import pytest
from aspect_based_sentiment_analysis import loads
from aspect_based_sentiment_analysis import ClassifierExample


def test_load_classifier_examples_laptops():
    examples = loads.load_classifier_examples(
        dataset='semeval',
        domain='laptops',
        test=False
    )
    assert len(examples) == 2313
    assert isinstance(examples[0], ClassifierExample)

    with pytest.raises(ValueError):
        loads.load_classifier_examples(dataset='mistake')
