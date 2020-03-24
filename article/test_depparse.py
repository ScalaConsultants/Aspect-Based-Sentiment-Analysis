# Model Performance
# https://stanfordnlp.github.io/stanza/performance.html
# https://github.com/UniversalDependencies/UD_English-EWT
# https://universaldependencies.org
# https://universaldependencies.org/u/dep/index.html

import os
from collections import Counter
from typing import List

import pytest
from stanza.models.common.doc import Sentence

import aspect_based_sentiment_analysis as absa
from article import depparse

EWT_CoNLL = 'data/UD_English-EWT/en_ewt-ud-train.conllu'


@pytest.fixture
def sentences():
    sentences = depparse.get_sentences(EWT_CoNLL)
    relations = [relation
                 for sentence in sentences
                 for head, relation, dependent in sentence.dependencies]
    counted_relations = Counter(relations)
    counted_relations.pop('root')
    counted_common_relations = counted_relations.most_common(n=20)
    assert counted_common_relations == [('punct', 23596), ('case', 17439),
                                        ('nsubj', 16258), ('det', 15754),
                                        ('advmod', 10958), ('obj', 10186),
                                        ('obl', 9154), ('amod', 9065),
                                        ('compound', 8197), ('conj', 7563),
                                        ('mark', 7479), ('nmod', 6905),
                                        ('cc', 6783), ('aux', 6511),
                                        ('cop', 4465), ('advcl', 3847),
                                        ('nmod:poss', 3674), ('xcomp', 3020),
                                        ('nummod', 2537), ('ccomp', 2388)]
    return sentences


@pytest.mark.skip
@pytest.mark.skipif(not os.path.isfile(EWT_CoNLL),
                    reason='Please download the EWT CoNLL train dataset.')
def test_filter_sentences(sentences: List[Sentence]):
    sent_1, sent_2, sent_3 = sentences = sentences[:3]
    is_in = lambda s, r: any(r == rel for head, rel, dep in s.dependencies)

    relation = 'ccomp'
    assert not is_in(sent_1, relation)
    assert not is_in(sent_2, relation)
    assert is_in(sent_3, relation)
    filtered_sentences = depparse.filter_sentences(sentences, relation)
    filtered_sentences = list(filtered_sentences)
    assert len(filtered_sentences) == 1

    relation = 'mark'
    assert not is_in(sent_1, relation)
    assert is_in(sent_2, relation)
    assert is_in(sent_3, relation)
    filtered_sentences = depparse.filter_sentences(sentences, relation)
    filtered_sentences = list(filtered_sentences)
    assert len(filtered_sentences) == 2


@pytest.mark.skip
@pytest.mark.skipif(not os.path.isfile(EWT_CoNLL),
                    reason='Please download the EWT CoNLL train dataset.')
def test_get_examples(sentences: List[Sentence]):
    sentence = sentences[0]
    relation = 'amod'
    direction = 'towards head'
    examples = depparse.get_examples(sentence, relation, direction)
    example_1, example_2 = examples = list(examples)
    assert len(examples) == 2
    assert example_1[0] == example_2[0]
    assert example_1[0] == ['Al', '-', 'Zaman', ':', 'American', 'forces',
                            'killed', 'Shaikh', 'Abdullah', 'al', '-', 'Ani',
                            ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in',
                            'the', 'town', 'of', 'Qaim', ',', 'near', 'the',
                            'Syrian', 'border', '.']

    tokens, x, y = example_1
    assert tokens[x] == 'American'
    assert tokens[y] == 'forces'
    tokens, x, y = example_2
    assert tokens[x] == 'Syrian'
    assert tokens[y] == 'border'

    relation = 'nmod'
    direction = 'towards dependent'
    examples = depparse.get_examples(sentence, relation, direction)
    example_1, example_2, example_3 = examples = list(examples)
    assert len(examples) == 3
    tokens, x, y = example_2
    assert tokens[x] == 'town'
    assert tokens[y] == 'Qaim'


# @pytest.mark.skip
@pytest.mark.skipif(not os.path.isfile(EWT_CoNLL),
                    reason='Please download the EWT CoNLL train dataset.')
def test_head_match(sentences: List[Sentence]):
    nlp = absa.pipeline('absa/classifier-rest-0.1',
                        output_attentions=True,
                        output_hidden_states=True)
    relation = 'det'
    direction = 'towards head'
    sentences = depparse.filter_sentences(sentences, relation)
    examples = [e for sentence in sentences
                for e in depparse.get_examples(sentence, relation, direction)]

    batches = absa.utils.batches(examples, batch_size=32)
    for batch in batches:
        texts = [' '.join(tokens) for tokens, x, y in batch]
