from abc import ABC
from abc import abstractmethod
from typing import Tuple

import transformers
import tensorflow as tf
from tensorflow.keras import layers


class ABSClassifier(ABC):
    """ The Aspect Based Sequence Classifier """

    @property
    @abstractmethod
    def language_model(self) -> tf.keras.Model:
        """ """

    @abstractmethod
    def call(self, *args, **kwargs):
        """ """

    @abstractmethod
    def call_extractor(self, *args, **kwargs):
        """ """

    @abstractmethod
    def call_classifier(self, *args, **kwargs):
        """ """


class BertABSCConfig(transformers.BertConfig):

    def __init__(self,
                 num_polarities: int = 3,
                 tags: Tuple[str] = tuple('BIEO'),
                 **kwargs):
        super().__init__(**kwargs)
        self.tags = tags
        self.num_tags = len(tags)
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
        self.extractor = layers.Dense(config.num_tags,
                                      kernel_initializer=initializer,
                                      name='extractor')
        self.classifier = layers.Dense(config.num_polarities,
                                       kernel_initializer=initializer,
                                       name='classifier')

    @property
    def language_model(self) -> transformers.TFBertForPreTraining:
        return self._language_model

    def call(self, token_ids: tf.Tensor, training=False, **bert_kwargs):
        # Extract aspects. There are potential few aspects in a single sequence.
        extractor_logits, *extractor_details = self.call_extractor(
            token_ids, training=training, **bert_kwargs
        )

        # Convert input sequences and recognized aspects to sequence pairs.
        # The second sequence describes a single aspect.
        classifier_inputs = self.build_auxiliary_sentence(token_ids, extractor_logits)

        # Classify sequence pairs.
        classifier_logits, *classifier_details = self.call_classifier(
            classifier_inputs, training=training, **bert_kwargs
        )
        return [extractor_logits, extractor_details, classifier_logits, classifier_details]

    def call_extractor(self, token_ids: tf.Tensor, training=False, **bert_kwargs):
        kwargs = dict(training=training, **bert_kwargs)
        outputs = self.language_model.bert(token_ids, **kwargs)
        sequence_output, pooled_output, *details = outputs
        sequence_output = self.dropout(sequence_output, training=training)
        logits = self.extractor(sequence_output)
        return [logits, *details]

    def call_classifier(self, token_ids: tf.Tensor, training=False, **bert_kwargs):
        kwargs = dict(training=training, **bert_kwargs)
        outputs = self.language_model.bert(token_ids, **kwargs)
        sequence_output, pooled_output, *details = outputs
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)
        return [logits, *details]

    @staticmethod
    def build_auxiliary_sentence(token_ids: tf.Tensor, logits: tf.Tensor):
        # TODO: implement
        return token_ids


class BertABSExtractorAndClassifier(BertABSClassifier):
    """ The BERT Aspect Based Sequence Extractor and Classifier Separated """
    def __init__(self, config: BertABSCConfig, **kwargs):
        super().__init__(config, **kwargs)
        LM = transformers.TFBertForPreTraining
        self._language_model = None     # Mask common LM
        self._language_model_extractor = LM(config, name='lm_extractor')
        self._language_model_classifier = LM(config, name='lm_classifier')

    @property
    def language_model_extractor(self) -> transformers.TFBertForPreTraining:
        return self._language_model_extractor

    @property
    def language_model_classifier(self) -> transformers.TFBertForPreTraining:
        return self._language_model_classifier

    def call_extractor(self, inputs, training=False, **bert_kwargs):
        kwargs = dict(training=training, **bert_kwargs)
        outputs = self.language_model_extractor.bert(inputs, **kwargs)
        sequence_output, pooled_output, *details = outputs
        sequence_output = self.dropout(sequence_output, training=training)
        logits = self.extractor(sequence_output)
        return [logits, *details]

    def call_classifier(self, inputs, training=False, **bert_kwargs):
        kwargs = dict(training=training, **bert_kwargs)
        outputs = self.language_model_classifier.bert(inputs, **kwargs)
        sequence_output, pooled_output, *details = outputs
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)
        return [logits, *details]
