import logging
from abc import ABC
from abc import abstractmethod
from typing import Tuple, Optional, Union

import numpy as np

import transformers
import tensorflow as tf
from tensorflow.keras import layers
from transformers.modeling_tf_utils import TFModelInputType

logger = logging.getLogger('absa.model')


class ABSClassifier(tf.keras.Model, ABC):
    """
    The model's aim is to classify the sentiment. The model contains the
    fine-tuned language model, which holds most parameters. The classifier
    itself is a tiny linear layer on top of a language model.

    We use the BERT language model, because we can benefit from the BERT's
    next-sentence prediction and formulate the task as the sequence-pair
    classification. Each example is described as one sequence in the format:
    "[CLS] text subtokens [SEP] aspect subtokens [SEP]". The relation between
    the text and aspect is encoded into the CLS token. The classifier just
    makes a linear transformation of the final special CLS token representation.
    The pipeline applies the softmax to get distribution over sentiment classes.

    Note how to train a model. We start with the original BERT version as a
    basis, and we divide the training into two stages. Firstly, due to the
    fact that the BERT is pretrained on dry Wikipedia texts, we wish to bias
    language model towards more informal language or a specific domain. To do
    so, we select texts close to the target domain and do the self-supervised
    **language model** post-training. The routine is the same as for the
    pre-training, but we need carefully set up optimization parameters.
    Secondly, we do regular supervised training. We train the whole model
    using a labeled dataset to classify a sentiment.

    Please note that the package contains the submodule `absa.training`. You
    can find there complete routines to tune or train either the language
    model or the classifier. Check out examples on the package website.

    References:
        [BERT: Pre-training of Deep Bidirectional Transformers for Language
        Understanding](https://arxiv.org/abs/1810.04805)
        [Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing
        Auxiliary Sentence](http://arxiv.org/abs/1903.09588)
        [BERT Post-Training for Review Reading Comprehension and Aspect-based
        Sentiment Analysis](http://arxiv.org/abs/1904.02232)
        [Adapt or Get Left Behind: Domain Adaptation through BERT Language
        Model Finetuning for Aspect-Target Sentiment Classification]
        (http://arxiv.org/abs/1908.11860)
    """

    @abstractmethod
    def call(
            self,
            input_ids: tf.Tensor,
            attention_mask: tf.Tensor = None,
            token_type_ids: tf.Tensor = None,
            training: bool = False,
            **bert_kwargs
    ) -> Tuple[tf.Tensor, Tuple[tf.Tensor, ...], Tuple[tf.Tensor, ...]]:
        """
        Perform the sentiment classification. We formulate the task as the
        sequence-pair classification. Each example is described as one
        sequence in the format:
            "[CLS] text subtokens [SEP] aspect subtokens [SEP]".

        Parameters
        ----------
        input_ids
            Indices of input sequence subtokens in the vocabulary.
        attention_mask
            Bool mask used to avoid performing attention on padding token
            indices in a batch (this is not related with masks from the
            language modeling task).
        token_type_ids
            Segment token indices to indicate first and second portions
            of the inputs, zeros and ones.
        training
            Whether to activate a dropout (True) during training or
            to de-activate them (False) for evaluation.
        bert_kwargs
            Auxiliary parameters which we forward directly to
            the **transformers** language model implementation.

        Returns
        -------
        logits
            The classifier final outputs.
        hidden_states
            Tuple of tensors: one for the output of the embeddings and one
            for the output of each layer.
        attentions
            Tuple of tensors: Attentions weights after the attention softmax,
            used to compute the weighted average in the self-attention heads.
        """


def force_to_return_details(kwargs: dict):
    """ Force a model to output attentions and hidden states due to the fixed
    definition of the output batch (the well-defined interface). """
    condition = not kwargs.get('output_attentions', False) or \
                not kwargs.get('output_hidden_states', False)
    if condition:
        logger.info('Model should output attentions and hidden states.')
    kwargs['output_attentions'] = True
    kwargs['output_hidden_states'] = True


class BertABSCConfig(transformers.BertConfig):

    def __init__(self, num_polarities: int = 3, **kwargs):
        force_to_return_details(kwargs)
        super().__init__(**kwargs)
        self.num_polarities = num_polarities


class BertABSClassifier(ABSClassifier, transformers.TFBertPreTrainedModel):

    def __init__(self, config: BertABSCConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.bert = transformers.TFBertMainLayer(
            config, name="bert"
        )
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
        input_ids: Optional[TFModelInputType] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[tf.Tensor, Tuple[tf.Tensor, ...], Tuple[tf.Tensor, ...]]:
        inputs = transformers.modeling_tf_utils.input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )
        outputs = self.bert(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)
        return logits, outputs.hidden_states, outputs.attentions
