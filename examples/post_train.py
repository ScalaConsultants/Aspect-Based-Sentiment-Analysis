import tensorflow as tf
import transformers
import aspect_based_sentiment_analysis as absa


def experiment(ID: int, source: str, epochs: int,
               base_model_name: str = 'bert-base-uncased'):
    # We use 4 GPUs for our experiments
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():

        config = absa.BertABSCConfig.from_pretrained(base_model_name)
        model = absa.BertABSClassifier.from_pretrained(base_model_name, config=config)
        tokenizer = transformers.BertTokenizer.from_pretrained(base_model_name)

        docs = absa.load_docs(source)
        document_database = absa.DocumentStore.from_iterable(docs)
        dataset = absa.LanguageModelDataset(
            document_database,
            tokenizer=tokenizer,
            batch_size=32,
            max_num_tokens=256,
            short_seq_prob=0.1,
            mlm_probability=0.15
        )

        # Language Model Post-training
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-8)
        absa.post_train(model.language_model, optimizer, dataset, epochs)

    model.save_pretrained(f'./models/post-train-{ID}')


if __name__ == '__main__':
    """ 
    The searching space:
    text sources: yelp-dataset / amazon-laptops
    
    """
    pass
