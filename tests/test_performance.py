import pytest
import numpy as np
import aspect_based_sentiment_analysis as absa
from aspect_based_sentiment_analysis.training import ConfusionMatrix


@pytest.mark.slow
def test_semeval_classification_restaurant():
    examples = absa.load_examples(
        dataset='semeval', domain='restaurant', test=True)
    nlp = absa.load('absa/classifier-rest-0.2')
    metric = ConfusionMatrix(num_classes=3)
    confusion_matrix = nlp.evaluate(examples, metric, batch_size=32)
    confusion_matrix = confusion_matrix.numpy()
    accuracy = np.diagonal(confusion_matrix).sum() / confusion_matrix.sum()
    assert round(accuracy, 4) >= .8517
    # The model is chosen based on the dev set. Out of curiosity, we've
    # verified (on the test set) top 5 models and the best performing is here:
    # 'absa/classifier-rest-0.2.1': .8732


@pytest.mark.slow
def test_semeval_classification_laptop():
    examples = absa.load_examples(
        dataset='semeval', domain='laptop', test=True)
    nlp = absa.load('absa/classifier-lapt-0.2')
    metric = ConfusionMatrix(num_classes=3)
    confusion_matrix = nlp.evaluate(examples, metric, batch_size=32)
    confusion_matrix = confusion_matrix.numpy()
    accuracy = np.diagonal(confusion_matrix).sum() / confusion_matrix.sum()
    assert round(accuracy, 4) >= .7978
    # The model is chosen based on the dev set. Out of curiosity, we've
    # verified (on the test set) top 5 models and the best performing is here:
    # 'absa/classifier-lapt-0.2.1': .8040


@pytest.mark.slow
@pytest.mark.timeout(20)  # The pipeline requires first 15s to initialize.
def test_inference():
    sentencier = absa.sentencizer()
    nlp = absa.load(text_splitter=sentencier)
    text = ("My wife and I and our 4 year old daughter stopped here "
            "Friday-Sunday. We arrived about midday and there was a queue to "
            "check in, which seemed to take ages to go down. Our check is was "
            "fairly quick but not sure why others in front of us took so "
            "long. Our room was \"ready\" although the sofa bed for our "
            "daughter hadn't been made.")
    aspects = ["reception", "bed", "service", "staff",
               "location", "public", "breakfast"]
    nlp(text, aspects)
