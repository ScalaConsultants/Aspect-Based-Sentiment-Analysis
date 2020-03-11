import pytest
import aspect_based_sentiment_analysis as absa


@pytest.mark.performance
def test_semeval_classification_restaurants():
    dataset = absa.load_classifier_examples(dataset='semeval',
                                            domain='restaurant',
                                            test=True)
    nlp = absa.pipeline('absa/classifier-rest-0.1')

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
