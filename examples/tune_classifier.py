import tensorflow as tf
import transformers
import aspect_based_sentiment_analysis as absa


def experiment(ID: int, source: str,  test_source: str, epochs: int,
               base_model_name: str, tokenizer_name: str = 'bert-base-uncased'):
    # We use 4 GPUs for our experiments
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():

        model = absa.BertABSClassifier.from_pretrained(base_model_name)
        tokenizer = transformers.BertTokenizer.from_pretrained(tokenizer_name)

        examples = absa.load_classifier_examples(source)
        test_examples = absa.load_classifier_examples(test_source)

        dataset = absa.ClassifierDataset.from_iterable(
            examples, tokenizer, batch_size=32
        )
        test_dataset = absa.ClassifierDataset.from_iterable(
            test_examples, tokenizer, batch_size=32
        )

        # Language Model Post-training
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-8)
        absa.tune_classifier(model, optimizer, dataset, epochs, test_dataset)

    model.save_pretrained(f'./models/tuned-classifier-{ID}')


if __name__ == '__main__':
    """
    The searching space:
    base_model_name = f'./models/post-train-{ID}'
    """
    pass
