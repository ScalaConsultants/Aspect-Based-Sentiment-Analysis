from abc import ABC
from abc import abstractmethod

import transformers
import tensorflow as tf
from tensorflow.keras import layers


class ABSClassifier(ABC):
    """ The Aspect Based Sequence Classifier """

    @property
    def language_model(self) -> tf.keras.Model:
        """ """

    @abstractmethod
    def call(self, *args, **kwargs):
        """ """


class BertABSCConfig(transformers.BertConfig):

    def __init__(self, num_polarities: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.num_polarities = num_polarities


class BertABSClassifier(ABSClassifier, transformers.TFBertPreTrainedModel):
    """ The BERT Aspect Based Sequence Classifier """

    def __init__(self, config: BertABSCConfig, **kwargs):
        super().__init__(config, **kwargs)
        LM = transformers.TFBertForPreTraining
        self._language_model = LM(config, name='language_model')
        initializer = transformers.modeling_tf_utils.get_initializer(
            config.initializer_range
        )
        self.dropout = layers.Dropout(config.hidden_dropout_prob)
        self.classifier = layers.Dense(config.num_polarities,
                                       kernel_initializer=initializer,
                                       name='classifier')

    @property
    def language_model(self) -> transformers.TFBertForPreTraining:
        return self._language_model

    def call(self, token_ids: tf.Tensor, training=False, **bert_kwargs):
        outputs = self.language_model.bert(token_ids, training=training,
                                           **bert_kwargs)
        sequence_output, pooled_output, *details = outputs
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)
        return [logits, *details]
