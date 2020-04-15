from abc import ABC
from abc import abstractmethod

import transformers
import tensorflow as tf
from tensorflow.keras import layers


class ABSClassifier(tf.keras.Model, ABC):
    """
    The Aspect Based Sequence Classifier

    - filling in the blanks (not generative)
    - use BERT because of the next-sentence prediction
    - the aspect based sentiment classification as the sequence-pair
    classification task
    - each sample contains text and aspect in one sequence
    - the format of "[CLS] sA [SEP] sB [SEP]" so the relation between two
    sequences
      is encoded into CLS representation.
    - the model's aim is to classify a sequence
    - linear layer on top of the language model
    - we use BERT language model
    - classifier has only a linear layer, linear transformation of final
    special CLS token
    - most parameters are in the language model, classifier is a tiny layer

    - the model training has three phases: a model predicts any part of its
    input for any observed part
      (a lot of feedback)

    self-supervised: more provided information / more constrains on parameters

    - 1) (self-supervised) the language model is pretrained on the Wikipedia
    corpus
    - 2) (self-supervised) the language model is fine-tuned to more specific,
    sentiment texts
    - 3) the classifier training supervised fashion (the language model is
    adjusted as well)
    """

    @abstractmethod
    def call(
            self,
            token_ids: tf.Tensor,
            attention_mask: tf.Tensor = None,
            token_type_ids: tf.Tensor = None,
            training: bool = False,
            **bert_kwargs
    ):
        """

        Parameters
        ----------
        token_ids
        attention_mask
        token_type_ids
        training
        bert_kwargs

        Returns
        -------

        """


class BertABSCConfig(transformers.BertConfig):

    def __init__(self, num_polarities: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.num_polarities = num_polarities


class BertABSClassifier(ABSClassifier, transformers.TFBertPreTrainedModel):

    def __init__(self, config: BertABSCConfig, **kwargs):
        super().__init__(config, **kwargs)
        LM = transformers.TFBertForPreTraining
        self.language_model = LM(config, name='language_model')
        initializer = transformers.modeling_tf_utils.get_initializer(
            config.initializer_range
        )
        self.dropout = layers.Dropout(config.hidden_dropout_prob)
        self.classifier = layers.Dense(
            config.num_polarities,
            kernel_initializer=initializer,
            name='classifier'
        )

    def call(
            self,
            token_ids: tf.Tensor,
            attention_mask: tf.Tensor = None,
            token_type_ids: tf.Tensor = None,
            training: bool = False,
            **bert_kwargs
    ):
        outputs = self.language_model.bert(
            inputs=token_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            training=training,
            **bert_kwargs
        )
        sequence_output, pooled_output, *details = outputs
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)
        return [logits, *details]
