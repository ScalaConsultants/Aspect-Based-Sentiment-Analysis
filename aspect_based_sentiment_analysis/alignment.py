from functools import partial
from typing import List
from typing import Tuple

import tensorflow as tf
import transformers
import numpy as np

from .data_types import TokenizedExample


def tokenize(
        tokenizer: transformers.BertTokenizer,
        text: str,
        aspect: str
) -> TokenizedExample:
    """ Tokenize the example, the pair of two raw strings (text, aspect).
    Moreover, we have to split tokens to subtokens using the **word-piece
    tokenizer**, according to the input format of the language model. We take
    care to do the alignment between tokens and subtokens for better
    interpretability. """
    basic_tokenizer = tokenizer.basic_tokenizer
    wordpiece_tokenizer = tokenizer.wordpiece_tokenizer

    text_tokens = basic_tokenizer.tokenize(text)
    cls = [tokenizer.cls_token]
    sep = [tokenizer.sep_token]
    aspect_tokens = basic_tokenizer.tokenize(aspect) if aspect else None
    tokens = cls + text_tokens + sep + aspect_tokens + sep \
        if aspect else cls + text_tokens + sep

    aspect_subtokens = get_subtokens(wordpiece_tokenizer, aspect_tokens)
    text_subtokens = get_subtokens(wordpiece_tokenizer, text_tokens)
    sub_tokens, alignment = make_alignment(wordpiece_tokenizer, tokens)

    example = TokenizedExample(
        text=text,
        text_tokens=text_tokens,
        text_subtokens=text_subtokens,
        aspect=aspect,
        aspect_tokens=aspect_tokens,
        aspect_subtokens=aspect_subtokens,
        tokens=tokens,
        subtokens=sub_tokens,
        alignment=alignment
    )
    return example


def get_subtokens(
        tokenizer: transformers.WordpieceTokenizer,
        tokens: List[str]
) -> List[str]:
    """ Split tokens into subtokens according to the input format of the
    language model. """
    split = tokenizer.tokenize
    return [sub_token for token in tokens for sub_token in split(token)]


def make_alignment(
        tokenizer: transformers.WordpieceTokenizer,
        tokens: List[str]
) -> Tuple[List[str], List[List[int]]]:
    """ Make the alignment between tokens and the subtokens. It is
    useful to interpret results or to understand the model reasoning. """
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


def merge_tensor(tensor: tf.Tensor, alignment: List[List[int]]) -> tf.Tensor:
    """ Merge input sub-token attentions into token attentions. """

    def aggregate(a, fun):
        n = len(alignment)
        new = np.zeros(n)
        for i in range(n):
            new[i] = fun(a[alignment[i]])
        return new

    # For attention _to_ a split-up word, we sum up the attention weights
    # over its tokens. For attention _from_ a split-up word, we take the mean
    # of the attention weights over its tokens. In other words, we take the
    # mean over rows, and sum over columns of split tokens according to the
    # alignment. Note that if we go along the axis, the aggregation
    # impacts to orthogonal dimension.
    x = tensor.numpy()
    attention_to = partial(aggregate, fun=np.mean)
    x = np.apply_along_axis(attention_to, 2, x)
    attention_from = partial(aggregate, fun=np.sum)
    x = np.apply_along_axis(attention_from, 3, x)
    x = tf.convert_to_tensor(x)
    return x
