import pytest
import numpy as np
import aspect_based_sentiment_analysis as absa


@pytest.mark.slow
def test_semeval_classification_restaurants():
    examples = absa.load_classifier_examples(dataset='semeval',
                                             domain='restaurant',
                                             test=True)
    nlp = absa.pipeline('absa/classifier-rest-0.1')

    # Quick entry validation
    text = ("We are great fans of Slack, but we wish the subscriptions "
            "were more accessible to small startups.")
    slack, price = nlp(text, aspects=['slack', 'price'])
    assert slack.sentiment == absa.Sentiment.positive
    assert price.sentiment == absa.Sentiment.negative

    metric = absa.ConfusionMatrix(num_classes=3)
    confusion_matrix = nlp.evaluate(examples, metric)
    accuracy = np.diagonal(confusion_matrix).sum() / confusion_matrix.sum()
    assert round(accuracy, 3) >= 0.86
