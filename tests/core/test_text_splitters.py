import pytest
from aspect_based_sentiment_analysis import text_splitters

# Skip tests if you do not download a model.
pytest.importorskip('en_core_web_sm')


def test_sentencizer():
    sentencizer = text_splitters.sentencizer()
    text = ("Obviously, it is not the most... robust solution "
            "but it'll do fine in most cases. "
            "This works for abbreviations e.g., for instance.")
    sentences = sentencizer(text)
    assert len(sentences) == 2
    sentence_1, sentence_2 = sentences
    assert sentence_1 == ("Obviously, it is not the most... robust solution "
                          "but it'll do fine in most cases.")
    assert sentence_2 == "This works for abbreviations e.g., for instance."
