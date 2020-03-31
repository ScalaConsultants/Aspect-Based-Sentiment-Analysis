from dataclasses import dataclass
from functools import partial
from typing import List
from typing import Tuple

import transformers
import numpy as np
import tensorflow as tf


@dataclass(frozen=True)
class Document:
    text: str
    text_tokens: List[str]
    tokens: List[str]
    sub_tokens: List[str]
    alignment: List[List[int]]
    aspect: str = None
    aspect_tokens: List[str] = None


@dataclass(frozen=True)
class DocumentBatch:
    token_ids: tf.Tensor
    attention_mask: tf.Tensor
    token_type_ids: tf.Tensor
    documents: List[Document]


def make_document(
        tokenizer: transformers.BertTokenizer,
        text: str, aspect: str = None
) -> Document:
    """ """
    basic_tokenizer = tokenizer.basic_tokenizer
    wordpiece_tokenizer = tokenizer.wordpiece_tokenizer

    text_tokens = basic_tokenizer.tokenize(text)
    cls = [tokenizer.cls_token]
    sep = [tokenizer.sep_token]
    aspect_tokens = basic_tokenizer.tokenize(aspect) if aspect else None
    tokens = cls + text_tokens + sep + aspect_tokens + sep \
        if aspect else cls + text_tokens + sep

    sub_tokens, alignment = make_alignment(wordpiece_tokenizer, tokens)
    template = Document(text, text_tokens, tokens, sub_tokens,
                        alignment, aspect, aspect_tokens)
    return template


def make_alignment(
        tokenizer: transformers.WordpieceTokenizer,
        tokens: List[str]
) -> Tuple[List[str], List[List[int]]]:
    """ """
    i = 0
    sub_tokens = []
    alignment = []
    for token in tokens:

        indices = []
        word_pieces = tokenizer.tokenize(token)
        for sub_token in word_pieces:
            indices.append(i)
            sub_tokens.append(sub_token)
            i += 1

        alignment.append(indices)
    return sub_tokens, alignment


def merge_input_attentions(
        attentions: np.ndarray,
        alignment: List[List[int]]
) -> np.ndarray:
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
    attentions = np.apply_along_axis(attention_to, 2, attentions)
    attention_from = partial(aggregate, fun=np.sum)
    attentions = np.apply_along_axis(attention_from, 3, attentions)
    return attentions
