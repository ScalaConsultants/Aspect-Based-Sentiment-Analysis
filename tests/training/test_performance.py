import pytest
import numpy as np
import aspect_based_sentiment_analysis as absa
from aspect_based_sentiment_analysis.training import ConfusionMatrix


@pytest.mark.slow
def test_semeval_classification_restaurants():
    examples = absa.load_classifier_examples(dataset='semeval',
                                             domain='restaurant',
                                             test=True)
    nlp = absa.load('absa/classifier-rest-0.1')

    metric = ConfusionMatrix(num_classes=3)
    confusion_matrix = nlp.evaluate(examples, metric, batch_size=32)
    accuracy = np.diagonal(confusion_matrix).sum() / confusion_matrix.sum()
    assert round(accuracy, 3) >= 0.86
