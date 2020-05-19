import pytest
import numpy as np
import aspect_based_sentiment_analysis as absa
from aspect_based_sentiment_analysis.training import ConfusionMatrix


@pytest.fixture
def nlp() -> absa.Pipeline:
    sentencier = absa.sentencizer()
    recognizer = absa.probing.AttentionGradientProduct()
    nlp = absa.load(text_splitter=sentencier, pattern_recognizer=recognizer)
    return nlp


@pytest.mark.slow
@pytest.mark.timeout(25)  # First 10s requires a pipeline to initialize.
def test_inference(nlp: absa.Pipeline):
    text = ("My wife and I and our 4 year old daughter stopped here "
            "Friday-Sunday. We arrived about midday and there was a queue to "
            "check in, which seemed to take ages to go down. Our check is was "
            "fairly quick but not sure why others in front of us took so "
            "long. Our room was \"ready\" although the sofa bed for our "
            "daughter hadn't been made.")
    aspects = ["reception", "bed", "service", "staff",
               "location", "public", "breakfast"]
    nlp(text, aspects)


@pytest.mark.slow
def test_semeval_classification_restaurants():
    examples = absa.load_examples(dataset='semeval',
                                  domain='restaurant',
                                  test=True)
    nlp = absa.load('absa/classifier-rest-0.1')

    metric = ConfusionMatrix(num_classes=3)
    confusion_matrix = nlp.evaluate(examples, metric, batch_size=32)
    confusion_matrix = confusion_matrix.numpy()
    accuracy = np.diagonal(confusion_matrix).sum() / confusion_matrix.sum()
    assert round(accuracy, 3) >= 0.86
