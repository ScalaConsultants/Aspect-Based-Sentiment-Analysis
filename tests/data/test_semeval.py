import io
from data.semeval import adapter
from aspect_based_sentiment_analysis import Label


def test_read_sentences():
    xml_text = """
    <sentences>
        <sentence id="1">test 1</sentence>
        <sentence id="2">test 2</sentence>
        <sentence id="3">test 3</sentence>
    </sentences>
    """
    file = io.StringIO(xml_text)
    sentences = adapter.read_sentences(file)
    assert len(sentences) == 3


def test_validate_sentence():
    xml_text = """
    <sentences>
        <sentence id="1">test 1</sentence>
        <sentence id="2">
            <text>test 2</text>
            <aspectTerms>
                <aspectTerm polarity="conflict"/>
            </aspectTerms>
        </sentence>
        <sentence id="3">
            <text>test 3</text>
            <aspectTerms>
                <aspectTerm polarity="neutral"/>
                <aspectTerm polarity="conflict"/>
            </aspectTerms>
        </sentence>
    </sentences>
    """
    file = io.StringIO(xml_text)
    sentences = adapter.read_sentences(file)
    stats = {'rejected': 0, 'conflicts': 0}
    validated_sentence, = adapter.validate_sentences(sentences, stats)
    assert validated_sentence.find('text').text == 'test 3'
    assert stats == {'rejected': 2, 'conflicts': 2}


def test_generate_classifier_examples():
    text = 'I charge it at night and skip taking the cord ' \
           'with me because of the good battery life.'
    xml_text = f"""
    <sentences>
        <sentence id="1">
            <text>{text}</text>
            <aspectTerms>
                <aspectTerm term="cord" polarity="neutral" from="41" to="45"/>
                <aspectTerm term="battery life" polarity="positive" from="74" to="86"/>
            </aspectTerms>
        </sentence>
    </sentences>
    """
    file = io.StringIO(xml_text)
    sentence, = adapter.read_sentences(file)
    examples = adapter.generate_classifier_examples(sentence)
    example_1, example_2 = examples
    assert example_1.aspect.name == 'cord'
    assert example_1.aspect.label == Label.neutral
    assert example_2.aspect.name == 'battery life'
    assert example_2.aspect.label == Label.positive
    assert example_1.text == example_2.text == text.lower()
