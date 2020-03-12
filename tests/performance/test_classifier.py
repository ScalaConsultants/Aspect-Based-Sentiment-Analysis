import pytest
import aspect_based_sentiment_analysis as absa


@pytest.mark.performance
def test_semeval_classification_restaurants():
    dataset = absa.load_classifier_examples(dataset='semeval',
                                            domain='restaurant',
                                            test=True)
    nlp = absa.pipeline('absa/classifier-rest-0.1')

    # Quick entry validation
    text = ("We are great fans of Slack, but we wish the subscriptions "
            "were more accessible to small startups.")
    prediction, = nlp(text, aspect_names=['Slack'])
    assert prediction.label == absa.Label.positive
    prediction, = nlp(text, aspect_names=['price'])
    assert prediction.label == absa.Label.negative

    results = []
    for example in dataset:
        prediction, = nlp.predict(
            example.text,
            aspect_names=[example.aspect.name]
        )
        results.append([example, prediction])

    is_correct = [example.aspect.label == prediction.label
                  for example, prediction in results]
    accuracy = sum(is_correct) / len(is_correct)
    assert round(accuracy, 3) >= 0.86

