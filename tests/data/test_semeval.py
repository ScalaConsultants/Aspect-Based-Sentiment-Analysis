import io
import os
import numpy as np
import pytest
from data.semeval import adapter
import aspect_based_sentiment_analysis as absa
from aspect_based_sentiment_analysis import Sentiment

# More details can be found here:
# https://github.com/davidsbatista/Aspect-Based-Sentiment-Analysis/tree/master/datasets/ABSA-SemEval2014
root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
data_dir = os.path.join(root_dir, 'data', 'semeval')
laptop_data = os.path.join(data_dir, 'Laptop_Train_v2.xml')
restaurant_data = os.path.join(data_dir, 'Restaurants_Train_v2.xml')
test_laptop_data = os.path.join(data_dir, 'Laptops_Test_Gold.xml')
test_restaurant_data = os.path.join(data_dir, 'Restaurants_Test_Gold.xml')


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
    assert example_1.aspect == 'cord'
    assert example_1.sentiment == Sentiment.neutral
    assert example_2.aspect == 'battery life'
    assert example_2.sentiment == Sentiment.positive
    assert example_1.text == example_2.text == text.lower()


@pytest.mark.skipif(not os.path.isfile(restaurant_data),
                    reason='Please download the restaurant train dataset.')
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
                              if example.sentiment == label)
    ratios = np.array([count(label) for label in Sentiment]) / len(examples)
    # The labels are in the order: [neutral, negative, positive]
    assert ratios.round(2).tolist() == [0.18, 0.22, 0.6]
    file_path = os.path.join(data_dir,
                             'classifier-semeval-restaurant-train.bin')
    absa.utils.save(examples, file_path)


@pytest.mark.skipif(not os.path.isfile(laptop_data),
                    reason='Please download the laptop train dataset.')
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
                              if example.sentiment == label)
    ratios = np.array([count(label) for label in Sentiment]) / len(examples)
    # The labels are in the order: [neutral, negative, positive]
    assert ratios.round(2).tolist() == [0.2, 0.37, 0.43]
    file_path = os.path.join(data_dir, 'classifier-semeval-laptop-train.bin')
    absa.utils.save(examples, file_path)


@pytest.mark.skipif(not os.path.isfile(test_restaurant_data),
                    reason='Please download the restaurant test dataset.')
def test_restaurant_test_dataset():
    with open(test_restaurant_data, 'r') as file:
        raw_sentences = adapter.read_sentences(file)
        assert len(raw_sentences) == 800

    stats = {'rejected': 0, 'conflicts': 0}
    sentences = adapter.validate_sentences(raw_sentences, stats)
    sentences = list(sentences)
    assert len(sentences) == 600
    assert stats['rejected'] == 200
    assert stats['conflicts'] == 14

    generate = adapter.generate_classifier_examples
    examples = [example
                for sentence in sentences
                for example in generate(sentence)]

    assert len(examples) == 1120
    # Check distribution of polarities
    count = lambda label: sum(True for example in examples
                              if example.sentiment == label)
    ratios = np.array([count(label) for label in Sentiment]) / len(examples)
    # The labels are in the order: [neutral, negative, positive]
    assert ratios.round(2).tolist() == [0.18, 0.18, 0.65]
    file_path = os.path.join(data_dir, 'classifier-semeval-restaurant-test.bin')
    absa.utils.save(examples, file_path)


@pytest.mark.skipif(not os.path.isfile(test_laptop_data),
                    reason='Please download the laptop test dataset.')
def test_laptop_test_dataset():
    with open(test_laptop_data, 'r') as file:
        raw_sentences = adapter.read_sentences(file)
        assert len(raw_sentences) == 800

    stats = {'rejected': 0, 'conflicts': 0}
    sentences = adapter.validate_sentences(raw_sentences, stats)
    sentences = list(sentences)
    assert len(sentences) == 411
    assert stats['rejected'] == 389
    assert stats['conflicts'] == 16

    generate = adapter.generate_classifier_examples
    examples = [example
                for sentence in sentences
                for example in generate(sentence)]

    assert len(examples) == 638
    # Check distribution of polarities
    count = lambda label: sum(True for example in examples
                              if example.sentiment == label)
    ratios = np.array([count(label) for label in Sentiment]) / len(examples)
    # The labels are in the order: [neutral, negative, positive]
    assert ratios.round(2).tolist() == [0.26, 0.2, 0.53]
    file_path = os.path.join(data_dir, 'classifier-semeval-laptop-test.bin')
    absa.utils.save(examples, file_path)
