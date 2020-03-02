import io
import os
import numpy as np
import pytest
from data.semeval import adapter
from aspect_based_sentiment_analysis import Label

root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
data_dir = os.path.join(root_dir, 'data', 'semeval')
laptop_data = os.path.join(data_dir, 'Laptop_Train_v2.xml')
restaurant_data = os.path.join(data_dir, 'Restaurants_Train_v2.xml')


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


@pytest.mark.skipif(not os.path.isfile(restaurant_data),
                    reason='Please download the restaurant train dataset to run a test.')
def test_restaurant_dataset():
    with open(restaurant_data, 'r') as file:
        raw_sentences = adapter.read_sentences(file)
        assert len(raw_sentences) == 3041

    stats = {'rejected': 0, 'conflicts': 0}
    sentences = adapter.validate_sentences(raw_sentences, stats)
    sentences = list(sentences)
    assert len(sentences) == 1978
    assert stats['rejected'] == 1063
    assert stats['conflicts'] == 91

    generate = adapter.generate_classifier_examples
    examples = [example
                for sentence in sentences
                for example in generate(sentence)]

    assert len(examples) == 3602
    # Check distribution of polarities
    count = lambda label: sum(True for example in examples
                              if example.aspect.label == label)
    ratios = np.array([count(label) for label in Label]) / len(examples)
    # The labels are in the order: [neutral, negative, positive]
    assert ratios.round(2).tolist() == [0.18, 0.22, 0.6]


@pytest.mark.skipif(not os.path.isfile(laptop_data),
                    reason='Please download the laptop train dataset to run a test.')
def test_laptop_dataset():
    with open(laptop_data, 'r') as file:
        raw_sentences = adapter.read_sentences(file)
        assert len(raw_sentences) == 3045

    stats = {'rejected': 0, 'conflicts': 0}
    sentences = adapter.validate_sentences(raw_sentences, stats)
    sentences = list(sentences)
    assert len(sentences) == 1462
    assert stats['rejected'] == 1583
    assert stats['conflicts'] == 45

    generate = adapter.generate_classifier_examples
    examples = [example
                for sentence in sentences
                for example in generate(sentence)]

    assert len(examples) == 2313
    # Check distribution of polarities
    count = lambda label: sum(True for example in examples
                              if example.aspect.label == label)
    ratios = np.array([count(label) for label in Label]) / len(examples)
    # The labels are in the order: [neutral, negative, positive]
    assert ratios.round(2).tolist() == [0.2, 0.37, 0.43]
