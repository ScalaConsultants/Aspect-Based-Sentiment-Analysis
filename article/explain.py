from typing import (
    List,
    Tuple,
    Callable,
    Iterable
)
from dataclasses import dataclass
from functools import partial

import transformers
import numpy as np
import tensorflow as tf

import aspect_based_sentiment_analysis as absa


@dataclass(frozen=True)
class Template:
    text: str
    aspect: str
    aspect_tokens: List[str]
    tokens: List[str]
    sub_tokens: List[str]
    alignment: List[List[int]]


def get_classifier_attentions(
        pipeline: str,
        domain: str
) -> Iterable[Tuple[Template, tf.Tensor, tf.Tensor]]:
    nlp = absa.pipeline(pipeline,
                        output_attentions=True,
                        output_hidden_states=True)
    examples = absa.load_classifier_examples(dataset='semeval',
                                             domain=domain,
                                             test=False)

    batches = absa.utils.batches(examples, batch_size=32)
    for batch in batches:
        pairs = [(e.text, e.aspect) for e in batch]
        with tf.GradientTape() as tape:
            encoded = nlp.tokenizer.batch_encode_plus(
                batch_text_or_text_pairs=pairs,
                add_special_tokens=True,
                pad_to_max_length='right',
                return_tensors='tf'
            )
            logits, hidden_states, attentions = nlp.model.call(
                token_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask'],
                token_type_ids=encoded['token_type_ids']
            )
            target_labels = tf.one_hot([e.sentiment for e in batch], depth=3)
            loss_value = absa.classifier_loss(target_labels, logits)
        d_attentions = tape.gradient(loss_value, attentions)

        # Stack a tensor tuple into a single multi-dim array:
        # [batch, layer, head, attention, attention]
        stack = lambda tensors: tf.transpose(tf.stack(tensors),
                                             [1, 0, 2, 3, 4]).numpy()
        attentions = stack(attentions)
        d_attentions = stack(d_attentions)
        templates = [make_template(nlp.tokenizer, pair) for pair in pairs]

        for i, template in enumerate(templates):
            alignment = template.alignment
            merge = merge_input_attentions
            α = merge(attentions[i], alignment)
            dα = merge(d_attentions[i], alignment)
            yield template, α, dα


def make_template(tokenizer: transformers.BertTokenizer,
                  pair: Tuple[str, str]) -> Template:
    text, aspect = pair
    basic_tokenizer = tokenizer.basic_tokenizer
    wordpiece_tokenizer = tokenizer.wordpiece_tokenizer

    text_tokens = basic_tokenizer.tokenize(text)
    aspect_tokens = basic_tokenizer.tokenize(aspect)
    cls = [tokenizer.cls_token]
    sep = [tokenizer.sep_token]
    tokens = cls + text_tokens + sep + aspect_tokens + sep

    i = 0
    sub_tokens = []
    alignment = []
    for token in tokens:

        indices = []
        word_pieces = wordpiece_tokenizer.tokenize(token)
        for sub_token in word_pieces:
            indices.append(i)
            sub_tokens.append(sub_token)
            i += 1

        alignment.append(indices)

    template = Template(text, aspect, aspect_tokens,
                        tokens, sub_tokens, alignment)
    return template


def merge_input_attentions(attentions: np.ndarray,
                           alignment: List[List[int]]) -> np.ndarray:
    """ Merge input sub-token attentions into token attentions. """

    def aggregate(a, fun):
        n = len(alignment)
        new = np.zeros(n)
        for i in range(n):
            new[i] = fun(a[alignment[i]])
        return new

    # For attention _to_ a split-up word, we sum up the attention weights
    # over its tokens. For attention _from_ a split-up word, we take the mean
    # of the attention weights over its tokens. Note, we switch aggregation
    # functions because if we go along the axis, the aggregation impacts to
    # orthogonal one.
    attention_to = partial(aggregate, fun=np.mean)
    α = np.apply_along_axis(attention_to, 2, attentions)
    attention_from = partial(aggregate, fun=np.sum)
    α = np.apply_along_axis(attention_from, 3, α)
    return α


def calculate_activation_means(α: np.ndarray,
                               template: Template,
                               pattern: Tuple[str, str]) -> np.ndarray:
    attention_from, attention_to = pattern
    tokens = template.tokens
    aspect_tokens = template.aspect_tokens

    i = get_mask(tokens, attention_from, aspect_tokens)
    j = get_mask(tokens, attention_to, aspect_tokens)

    # Unfortunately, we have to perform the slicing operation
    # in two separate steps.
    interested = α[..., i, :][..., j]
    means = np.mean(interested, axis=(2, 3))
    return means


def get_mask(tokens: List[str], option: str,
             aspect_tokens: List[str]) -> np.ndarray:
    rule = get_rule(option, aspect_tokens)
    mask = np.array([rule(t) for t in tokens])
    return mask


def get_rule(option: str, aspect_tokens: List[str]) -> Callable[[str], bool]:
    rules = {
        'ALL': lambda token: True,
        'NON-SPECIAL': lambda token:
        token not in ['[CLS]', '[SEP]', *aspect_tokens],
        'CLS': lambda token: token == '[CLS]',
        'SEP': lambda token: token == '[SEP]',
        'ASPECT': lambda token: token in aspect_tokens
    }
    return rules[option]
