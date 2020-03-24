from dataclasses import dataclass
from typing import List
from typing import Iterable
from typing import Tuple

import numpy as np
import stanza
import tensorflow as tf
import transformers
from stanza.models.common.doc import Sentence
from stanza.utils.conll import CoNLL

from article.explain import make_alignment
from aspect_based_sentiment_analysis import BertPipeline


@dataclass(frozen=True)
class Document:
    text: str
    text_tokens: str
    tokens: List[str]
    sub_tokens: List[str]
    alignment: List[List[int]]


def get_language_model_attentions(nlp: BertPipeline,
                                  texts: List[str]) -> Iterable[np.ndarray]:
    encoded = nlp.tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        pad_to_max_length='right',
        return_tensors='tf'
    )
    outputs = nlp.model.language_model.bert(
        token_ids=encoded['input_ids'],
        attention_mask=encoded['attention_mask'],
        token_type_ids=encoded['token_type_ids']
    )
    attentions = outputs[-1]
    # Stack a tensor tuple into a single multi-dim array:
    # [batch, layer, head, attention, attention]
    stack = lambda tensors: tf.transpose(tf.stack(tensors),
                                         [1, 0, 2, 3, 4]).numpy()
    attentions = stack(attentions)
    documents = [make_document(nlp.tokenizer, text) for text in texts]
    yield documents, attentions


def make_document(tokenizer: transformers.BertTokenizer, text: str) -> Document:
    basic_tokenizer = tokenizer.basic_tokenizer
    wordpiece_tokenizer = tokenizer.wordpiece_tokenizer

    text_tokens = basic_tokenizer.tokenize(text)
    cls = [tokenizer.cls_token]
    sep = [tokenizer.sep_token]
    tokens = cls + text_tokens + sep

    sub_tokens, alignment = make_alignment(wordpiece_tokenizer, tokens)
    template = Document(text, text_tokens, tokens, sub_tokens, alignment)
    return template


def get_examples(sentence: Sentence,
                 relation: str,
                 direction: str = 'towards head') -> Tuple[List[str], int, int]:
    """ Directions: `towards head` or `towards dependent` """
    tokens = [token.text for token in sentence.tokens]
    for head, rel, dep in sentence.dependencies:
        if rel == relation:
            x, y = int(dep.id) - 1, int(head.id) - 1  # Token indices
            x, y = (y, x) if direction == 'towards dependent' else (x, y)
            yield tokens, x, y


def filter_sentences(sentences: List[Sentence],
                     relation: str) -> Iterable[Sentence]:
    for sentence in sentences:
        is_in = any(relation == rel for head, rel, dep in sentence.dependencies)
        if is_in:
            yield sentence


def get_sentences(fname: str, limit: int = None) -> List[Sentence]:
    with open(fname, 'r') as f:
        conll = CoNLL.load_conll(f)
    sentences = CoNLL.convert_conll(conll)
    sentences = sentences[:limit]
    document = stanza.Document(sentences)
    return document.sentences
