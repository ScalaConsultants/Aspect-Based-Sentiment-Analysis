from functools import partial
from typing import List
from typing import Tuple

import transformers
import numpy as np

from .data_types import AspectSpan


def make_aspect_span(
        tokenizer: transformers.BertTokenizer,
        text: str,
        aspect: str = None  # None for the experiment purposes.
) -> AspectSpan:
    """ Make the `AspectSpan` from the raw text/aspect pair. As for most NLP
    tasks, we need to tokenize raw strings at the very beginning. Besides,
    we have to split tokens to sub_tokens using the *word-piece tokenizer*,
    according to the input format of the language model. We take care to do
    the alignment between them for better interpretability. """
    basic_tokenizer = tokenizer.basic_tokenizer
    wordpiece_tokenizer = tokenizer.wordpiece_tokenizer

    text_tokens = basic_tokenizer.tokenize(text)
    cls = [tokenizer.cls_token]
    sep = [tokenizer.sep_token]
    aspect_tokens = basic_tokenizer.tokenize(aspect) if aspect else None
    tokens = cls + text_tokens + sep + aspect_tokens + sep \
        if aspect else cls + text_tokens + sep

    sub_tokens, alignment = make_alignment(wordpiece_tokenizer, tokens)
    aspect_span = AspectSpan(
        text=text,
        text_tokens=text_tokens,
        aspect=aspect,
        aspect_tokens=aspect_tokens,
        tokens=tokens,
        sub_tokens=sub_tokens,
        alignment=alignment
    )
    return aspect_span


def make_alignment(
        tokenizer: transformers.WordpieceTokenizer,
        tokens: List[str]
) -> Tuple[List[str], List[List[int]]]:
    """ Make the alignment between tokens and the sub-tokens. It is
    useful to interpret results or understand the model reasoning. """
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
